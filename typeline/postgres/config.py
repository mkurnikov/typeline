import logging
import sys
from typing import Iterable, Optional, List, Union

import psycopg2
from psycopg2.extras import LoggingConnection

from typeline.config import DefaultConfig
from typeline.db.base import CallTraceStore, CallTraceStoreLogger, CallTraceThunk
from typeline.encoding import serialize_traces, arg_types_to_json, CallTraceRow, ClassPropsTraceRow
from typeline.tracing import CallTrace, ClassPropsTrace


BATCH_SIZE = 30000
REMOVE_DUPLICATES_BATCHES_SIZE = 3
VACUUM_DUP_REMOVALS = 2
FUNCTION_CALLS_TABLE = 'monkeytype_call_traces'
CLASS_PROPERTIES_TABLE = 'class_props_traces'
N_DEAD_TUPLES_FOR_VACUUM = 1_000_000

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logger = logging.getLogger(__name__)


class MyLoggingConnection(LoggingConnection):
    def filter(self, msg, curs):
        return msg.decode()[:1000]


_REMOVE_DUPLICATES_QUERY = """
DELETE FROM {table}
WHERE id IN (SELECT id
             FROM (SELECT
                     id,
                     row_number()
                     OVER (
                       partition BY {columns}
                       ORDER BY id ) AS rnum
                   FROM {table}) t
             WHERE t.rnum > 1);
"""
REMOVE_DUPLICATES_FUNCTION_TABLE = _REMOVE_DUPLICATES_QUERY.format(table=FUNCTION_CALLS_TABLE,
                                                                   columns='module, qualname, arg_types, return_type, yield_type')
REMOVE_DUPLICATES_FROM_CLASS_TABLE = _REMOVE_DUPLICATES_QUERY.format(table=CLASS_PROPERTIES_TABLE,
                                                                     columns='module, qualname, props')

INSERT_FUNC_CALLS_QUERY = """
INSERT INTO {table}(module, qualname, arg_types, return_type, yield_type)
VALUES %s
""".format(table=FUNCTION_CALLS_TABLE)

INSERT_CLASS_PROPS_QUERY = """
INSERT INTO {table}(module, qualname, props)
VALUES %s
""".format(table=CLASS_PROPERTIES_TABLE)

FILTER_QUERY = """
SELECT module, qualname, arg_types, return_type, yield_type
FROM monkeytype_call_traces
WHERE module = %s
"""

QueryValue = Union[str, int]


def make_filter_query(module):
    query = FILTER_QUERY
    values = [module]
    # if qualname is not None:
    #     query += "AND qualname LIKE %s || '%'"
    #     values += [qualname]

    query = query.strip() + ';'
    return query, values

# def make_class_filter_query(module: str, qualname: str) ->
CLASS_FILTER_QUERY = """
SELECT module, qualname, props
FROM class_props_traces
WHERE module = %s
"""


def create_call_trace_table(connection: LoggingConnection):
    with connection.cursor() as cursor:
        query = """
CREATE TABLE IF NOT EXISTS {func_table} (
    id          SERIAL PRIMARY KEY ,
    module      TEXT,
    qualname    TEXT,
    arg_types   JSONB,
    return_type JSONB,
    yield_type  JSONB
);

CREATE TABLE IF NOT EXISTS {class_table} (
    id SERIAL PRIMARY KEY ,
    module TEXT,
    qualname TEXT,
    props JSONB
)
""".format(func_table=FUNCTION_CALLS_TABLE,
           class_table=CLASS_PROPERTIES_TABLE)

        cursor.execute(query)
    connection.commit()


class PostgresStore(CallTraceStore):
    def __init__(self, conn, autocommit_conn, table: str) -> None:
        self.conn = conn
        self.autocommit_conn = autocommit_conn
        self.table = table

    def add(self, traces: Iterable[CallTrace]):
        values = []
        for row in serialize_traces(traces):
            values.append((row.module, row.qualname, row.arg_types,
                           row.return_type, row.yield_type))

        with self.conn.cursor() as cursor:
            psycopg2.extras.execute_values(cursor, sql=INSERT_FUNC_CALLS_QUERY, argslist=values)
        self.conn.commit()

    def add_classes(self, class_traces: Iterable[ClassPropsTrace]) -> None:
        values = []
        for class_trace in class_traces:
            module = class_trace.module
            qualname = class_trace.qualname
            self_params = arg_types_to_json(class_trace.class_props or {})
            values.append((module, qualname, self_params))

        with self.conn.cursor() as cursor:
            psycopg2.extras.execute_values(cursor, sql=INSERT_CLASS_PROPS_QUERY, argslist=values)
        self.conn.commit()

    def remove_duplicates(self):
        with self.conn.cursor() as cursor:
            cursor.execute(REMOVE_DUPLICATES_FUNCTION_TABLE)
            cursor.execute(REMOVE_DUPLICATES_FROM_CLASS_TABLE)
        self.conn.commit()

    def vacuum(self):
        self.conn.autocommit = True
        with self.conn.cursor() as cursor:
            cursor.execute('SELECT sum(n_dead_tup) FROM pg_stat_user_tables')
            num_dead_tuples = cursor.fetchone()[0]
            if num_dead_tuples > N_DEAD_TUPLES_FOR_VACUUM:
                cursor.execute('VACUUM;')
        self.conn.autocommit = False

    @classmethod
    def make_store(cls, connection_data: dict, log_queries: bool, skip_init=False) -> 'PostgresStore':
        if log_queries:
            conn = psycopg2.connect(**connection_data, connection_factory=MyLoggingConnection)
            conn.initialize(logger)
        else:
            conn = psycopg2.connect(**connection_data)

        autocommit_connection = psycopg2.connect(**connection_data, connection_factory=MyLoggingConnection)
        autocommit_connection.initialize(logger)
        autocommit_connection.autocommit = True

        if not skip_init:
            create_call_trace_table(conn)

        return cls(conn, autocommit_connection, FUNCTION_CALLS_TABLE)

    def filter(
            self,
            module: str,
            limit: int = 2000
    ) -> List[CallTraceThunk]:
        filter_query, params = make_filter_query(module)
        with self.conn.cursor() as cursor:
            cursor.execute(filter_query, tuple(params))

            trace_rows = []
            for row in cursor.fetchall():
                trace_rows.append(CallTraceRow(*row))

            return trace_rows

    def extract_class_props(self, module) -> List[ClassPropsTraceRow]:
        with self.conn.cursor() as cursor:
            cursor.execute(CLASS_FILTER_QUERY, (module,))

            class_trace_rows = []
            for row in cursor.fetchall():
                class_trace_rows.append(ClassPropsTraceRow(*row))

            return class_trace_rows

    def list_modules(self, prefix=None):
        with self.conn.cursor() as cursor:
            query = """
SELECT DISTINCT module 
FROM {table} 
WHERE module ILIKE %s
""".format(table=FUNCTION_CALLS_TABLE)

            prefix = (prefix or '') + '%'
            cursor.execute(query, (prefix,))
            return [row[0] for row in cursor.fetchall() if row[0]]


class PostgresLogger(CallTraceStoreLogger):
    store: PostgresStore

    def __init__(self, store: CallTraceStore, relevant_modules: Optional[List[str]] = None) -> None:
        super().__init__(store)
        self.num_of_traces = 0
        self.num_of_batches = 0
        self.num_of_duplicate_removals = 0
        self.relevant_modules = relevant_modules

    def _should_be_traced(self, trace: CallTrace):
        if not self.relevant_modules:
            return True

        if trace.func.__module__:
            for module in self.relevant_modules:
                return trace.func.__module__.startswith(module + '.') \
                       or trace.func.__module__ == module

    def log(self, trace: CallTrace):
        if self._should_be_traced(trace):
            self.traces.append(trace)
            self.num_of_traces += 1

        if self.num_of_traces > BATCH_SIZE:
            self.flush()
            self.num_of_batches += 1
            self.num_of_traces = 0

        if self.num_of_batches >= REMOVE_DUPLICATES_BATCHES_SIZE:
            self.store.remove_duplicates()
            self.num_of_duplicate_removals += 1
            self.num_of_batches = 0

        if self.num_of_duplicate_removals >= VACUUM_DUP_REMOVALS:
            self.store.vacuum()
            self.num_of_duplicate_removals = 0


class PostgresClassLogger(CallTraceStoreLogger):
    def __init__(self, store: CallTraceStore, relevant_modules: Optional[List[str]] = None) -> None:
        super().__init__(store)
        self.class_traces: List[ClassPropsTrace] = []

        self.num_of_traces = 0
        self.num_of_batches = 0
        self.num_of_duplicate_removals = 0
        self.relevant_modules = relevant_modules

    def _should_be_traced(self, trace: ClassPropsTrace):
        if not self.relevant_modules:
            return True

        if trace.module:
            for root_module in self.relevant_modules:
                return trace.module.startswith(root_module + '.') or trace.module == root_module

    def log(self, trace: ClassPropsTrace):
        if self._should_be_traced(trace):
            self.class_traces.append(trace)
            self.num_of_traces += 1

        if self.num_of_traces > BATCH_SIZE:
            self.flush()
            self.num_of_batches += 1
            self.num_of_traces = 0

    def flush(self):
        self.store.add_classes(self.class_traces)
        self.class_traces = []


class PostgresConfig(DefaultConfig):
    def __init__(self,
                 connection_data,
                 skip_private_methods=True,
                 skip_private_properties=True,
                 log_queries=False,
                 relevant_modules: Optional[List[str]] = None):
        self.connection_data = connection_data
        self.skip_private_methods = skip_private_methods
        self.skip_private_properties = skip_private_properties
        self.log_queries = log_queries
        self.relevant_modules = relevant_modules

    def trace_logger(self):
        return PostgresLogger(self.trace_store(),
                              relevant_modules=self.relevant_modules)

    def class_trace_logger(self):
        return PostgresClassLogger(self.trace_store(skip_init=True),
                                   relevant_modules=self.relevant_modules)

    def trace_store(self, skip_init=False) -> PostgresStore:
        return PostgresStore.make_store(connection_data=self.connection_data,
                                        log_queries=self.log_queries,
                                        skip_init=skip_init)

    # def type_rewriter(self):
    #     from monkeytype import rewriters
