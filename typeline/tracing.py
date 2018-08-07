# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import inspect
import logging
import opcode
import random
import sys
from abc import (
    ABCMeta,
    abstractmethod,
)
from contextlib import contextmanager
from types import (
    CodeType,
    FrameType,
)
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Optional,
    Union,
    cast,
    TYPE_CHECKING, List)

if TYPE_CHECKING:
    from typeline.postgres.config import PostgresConfig

try:
    from django.utils.functional import cached_property  # type: ignore
except ImportError:
    cached_property = None

from typeline.typing import get_type
from typeline.util import get_func_fqname


logger = logging.getLogger(__name__)


class ClassPropsTrace:
    def __init__(self,
                 module: str,
                 qualname: str,
                 class_props: Dict[str, type]):
        self.module = module
        self.qualname = qualname
        self.class_props = class_props

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.module, self.qualname, frozenset(self.class_props.items())))


class CallTrace:
    """CallTrace contains the types observed during a single invocation of a function"""

    def __init__(
            self,
            func: Callable,
            arg_types: Dict[str, type],
            return_type: Optional[type] = None,
            yield_type: Optional[type] = None
    ) -> None:
        """
        Args:
            func: The function where the trace occurred
            arg_types: The collected argument types
            return_type: The collected return type. This will be None if the called function returns
                due to an unhandled exception. It will be NoneType if the function returns the value None.
            yield_type: The collected yield type. This will be None if the called function never
                yields. It will be NoneType if the function yields the value None.
        """
        self.func = func
        self.arg_types = arg_types
        self.return_type = return_type
        self.yield_type = yield_type

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __repr__(self) -> str:
        return "CallTrace(%s, %s, %s, %s)" % (self.func, self.arg_types, self.return_type, self.yield_type)

    def __hash__(self) -> int:
        return hash((self.func, frozenset(self.arg_types.items()), self.return_type, self.yield_type))

    def add_yield_type(self, typ: type) -> None:
        if self.yield_type is None:
            self.yield_type = typ
        else:
            self.yield_type = Union[self.yield_type, typ]

    @property
    def funcname(self) -> str:
        return get_func_fqname(self.func)


class CallTraceLogger(metaclass=ABCMeta):
    """Log and store/print records collected by a CallTracer."""

    @abstractmethod
    def log(self, trace: CallTrace) -> None:
        """Log a single call trace."""
        pass

    def flush(self) -> None:
        """Flush all logged traces to output / database.

        Not an abstractmethod because it's OK to leave it as a no-op; for very
        simple loggers it may not be necessary to batch-flush traces, and `log`
        can handle everything.
        """
        pass


class ClassPropsTraceLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, trace: ClassPropsTrace) -> None:
        """Log a single call trace."""
        pass

    def flush(self) -> None:
        """Flush all logged traces to output / database.

        Not an abstractmethod because it's OK to leave it as a no-op; for very
        simple loggers it may not be necessary to batch-flush traces, and `log`
        can handle everything.
        """
        pass


def get_func_in_mro(obj: Any, code: CodeType) -> Optional[Callable]:
    """Attempt to find a function in a side-effect free way.

    This looks in obj's mro manually and does not invoke any descriptors.
    """
    # FunctionType is incompatible with Callable
    # https://github.com/python/typeshed/issues/1378
    val = inspect.getattr_static(obj, code.co_name, None)
    if val is None:
        return None
    if isinstance(val, (classmethod, staticmethod)):
        cand = cast(Callable, val.__func__)
    elif isinstance(val, property) and (val.fset is None) and (val.fdel is None):
        cand = cast(Callable, val.fget)
    elif cached_property and isinstance(val, cached_property):
        cand = cast(Callable, val.func)
    else:
        cand = cast(Callable, val)
    return _has_code(cand, code)


def _has_code(func: Optional[Callable], code: CodeType) -> Optional[Callable]:
    while func is not None:
        func_code = getattr(func, '__code__', None)
        if func_code is code:
            return func
        # Attempt to find the decorated function
        func = getattr(func, '__wrapped__', None)
    return None


def get_func(frame: FrameType) -> Optional[Callable]:
    """Return the function whose code object corresponds to the supplied stack frame."""
    code = frame.f_code
    if code.co_name is None:
        return None

    # First, try to find the function in globals
    cand = frame.f_globals.get(code.co_name, None)
    func = _has_code(cand, code)

    # If that failed, as will be the case with class and instance methods, try
    # to look up the function from the first argument. In the case of class/instance
    # methods, this should be the class (or an instance of the class) on which our
    # method is defined.
    if func is None and code.co_argcount >= 1:
        first_arg = frame.f_locals.get(code.co_varnames[0])
        func = get_func_in_mro(first_arg, code)

    # If we still can't find the function, as will be the case with static methods,
    # try looking at classes in global scope.
    if func is None:
        for v in frame.f_globals.values():
            if not isinstance(v, type):
                continue
            func = get_func_in_mro(v, code)
            if func is not None:
                break
    return func


RETURN_VALUE_OPCODE = opcode.opmap['RETURN_VALUE']
YIELD_VALUE_OPCODE = opcode.opmap['YIELD_VALUE']

# A CodeFilter is a predicate that decides whether or not a the call for the
# supplied code object should be traced.
CodeFilter = Callable[[CodeType], bool]

EVENT_CALL = 'call'
EVENT_RETURN = 'return'
SUPPORTED_EVENTS = {EVENT_CALL, EVENT_RETURN}


def is_method_of_class(frame: FrameType) -> bool:
    return 'self' in frame.f_locals


def is_private_function(code: CodeType) -> bool:
    return code.co_name.startswith('_') and not code.co_name.startswith('__')


class CallTracer:
    """CallTracer captures the concrete types involved in a function invocation.

    On a per function call basis, CallTracer will record the types of arguments
    supplied, the type of the function's return value (if any), and the types
    of values yielded by the function (if any). It emits a CallTrace object
    that contains the captured types when the function returns.

    Use it like so:

        sys.setprofile(CallTracer(MyCallLogger()))

    """

    def __init__(
            self,
            logger: CallTraceLogger,
            class_logger: ClassPropsTraceLogger,
            code_filter: Optional[CodeFilter] = None,
            sample_rate: Optional[int] = None,
            skip_private_methods: bool = True,
            skip_private_properties: bool = True,
            acceptable_modules: Optional[List[str]] = None
    ) -> None:
        self.logger = logger
        self.class_logger = class_logger
        self.traces: Dict[FrameType, CallTrace] = {}
        self.sample_rate = sample_rate
        self.cache: Dict[CodeType, Optional[Callable]] = {}
        self.should_trace = code_filter

        self.mro_replacements: Dict[type, type] = {}

        # self.class_traces: Dict[str, List[ClassPropsTrace]] = defaultdict(list)
        # self.self_params: Dict[FrameType, Dict[str, Any]] = {}
        # self.callables = defaultdict(list)

        self.skip_private_methods = skip_private_methods
        self.skip_private_properties = skip_private_properties

    def _get_func(self, frame: FrameType) -> Optional[Callable]:
        code = frame.f_code
        if code not in self.cache:
            self.cache[code] = get_func(frame)
        return self.cache[code]

    def handle_call(self, frame: FrameType) -> None:
        if self.sample_rate and random.randrange(self.sample_rate) != 0:
            return

        func = self._get_func(frame)
        if func is None:
            return

        code = frame.f_code
        if self.skip_private_methods and is_private_function(code):
            return

        # I can't figure out a way to access the value sent to a generator via
        # send() from a stack frame.
        if code.co_code[frame.f_lasti] == YIELD_VALUE_OPCODE:
            return

        arg_names = code.co_varnames[0:code.co_argcount]
        arg_types = {}
        for name in arg_names:
            if name in frame.f_locals:
                arg_value = frame.f_locals[name]
                arg_type = get_type(arg_value)
                # if arg_type == Callable:
                #     full_qualname = ':'.join([arg_value.__module__, arg_value.__qualname__])
                #     self.callables[full_qualname].append((name, arg_types))
                arg_types[name] = arg_type

        self.traces[frame] = CallTrace(func, arg_types)

    def handle_return(self, frame: FrameType, arg: Any) -> None:
        # In the case of a 'return' event, arg contains the return value, or
        # None, if the block returned because of an unhandled exception. We
        # need to distinguish the exceptional case (not a valid return type)
        # from a function returning (or yielding) None. In the latter case, the
        # the last instruction that was executed should always be a return or a
        # yield.

        typ = get_type(arg)
        last_opcode = frame.f_code.co_code[frame.f_lasti]
        trace = self.traces.get(frame)
        if trace is None:
            return
        elif last_opcode == YIELD_VALUE_OPCODE:
            trace.add_yield_type(typ)
        else:
            if last_opcode == RETURN_VALUE_OPCODE:
                trace.return_type = typ

            if is_method_of_class(frame):
                parent_obj = frame.f_locals['self']

                module = None
                try:
                    module = parent_obj.__class__.__module__
                except Exception:
                    pass

                qualname = None
                try:
                    qualname = parent_obj.__class__.__qualname__
                except Exception:
                    pass

                props = {}
                for key, val in list(parent_obj.__dict__.items()):
                    if self.skip_private_properties and key.startswith('_'):
                        continue
                    props[key] = get_type(val)

                class_trace = ClassPropsTrace(module, qualname, props)
                # logging.critical(class_trace)
                self.class_logger.log(class_trace)

            # func = self._get_func(frame)
            # full_qualname = ':'.join([func.__module__, func.__qualname__])
            # if full_qualname in self.callables:
            #     for name, arg_types in self.callables[full_qualname]:
            #         arg_types[name] = Callable[[*(trace.arg_types.values())], trace.return_type]
            #
            #     del self.callables[full_qualname]

            # logging.critical(trace)
            del self.traces[frame]
            self.logger.log(trace)

    def __call__(self, frame: FrameType, event: str, arg: Any) -> 'CallTracer':
        code = frame.f_code
        if (
                event not in SUPPORTED_EVENTS or
                code.co_name == 'trace_types' or
                self.should_trace and not self.should_trace(code)
        ):
            return self

        try:
            if event == EVENT_CALL:
                self.handle_call(frame)
            elif event == EVENT_RETURN:
                self.handle_return(frame, arg)
            else:
                logger.error("Cannot handle event %s", event)
        except Exception:
            logger.exception("Failed collecting trace")
        return self


@contextmanager
def trace_calls(
        logger: CallTraceLogger,
        code_filter: Optional[CodeFilter] = None,
        sample_rate: Optional[int] = None,
) -> Iterator[None]:
    """Enable call tracing for a block of code"""
    old_trace = sys.getprofile()
    sys.setprofile(CallTracer(logger, code_filter, sample_rate))
    try:
        yield
    finally:
        sys.setprofile(old_trace)
        logger.flush()


@contextmanager
def trace_calls_init_with_config(
        config: 'PostgresConfig'
) -> Iterator[None]:
    """Enable call tracing for a block of code"""
    old_trace = sys.getprofile()
    trace_logger = config.trace_logger()
    class_trace_logger = config.class_trace_logger()

    sys.setprofile(CallTracer(logger=trace_logger,
                              class_logger=class_trace_logger,
                              code_filter=config.code_filter(),
                              sample_rate=config.sample_rate(),
                              skip_private_properties=config.skip_private_properties,
                              skip_private_methods=config.skip_private_methods))

    if config.skip_private_properties:
        print('Skipping private properties')
    if config.skip_private_methods:
        print('Skipping private methods')

    try:
        yield
    finally:
        sys.setprofile(old_trace)
        trace_logger.flush()
        if hasattr(trace_logger, 'remove_duplicates'):
            trace_logger.remove_duplicates()

        class_trace_logger.flush()
        if hasattr(trace_logger, 'remove_duplicates'):
            trace_logger.remove_duplicates()
