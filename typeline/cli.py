# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import collections
import os
import os.path
import runpy
import sys
from typing import (
    IO,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

from typeline import trace
from typeline.config import Config
from typeline.exceptions import MonkeyTypeError
from typeline.postgres.config import PostgresConfig
from typeline.stubs import (
    Stub,
    build_module_stubs_from_traces,
)
from typeline.tracing import CallTrace
from typeline.typing import NoOpRewriter
from typeline.util import get_name_in_module


if TYPE_CHECKING:
    # This is not present in Python 3.6.1, so not safe for runtime import
    from typing import NoReturn  # noqa


def module_path(path: str) -> Tuple[str, Optional[str]]:
    """Parse <module>[:<qualname>] into its constituent parts."""
    parts = path.split(':', 1)
    module = parts.pop(0)
    qualname = parts[0] if parts else None
    if os.sep in module:  # Smells like a path
        raise argparse.ArgumentTypeError(
            f'{module} does not look like a valid Python import path'
        )

    return module, qualname


def module_path_with_qualname(path: str) -> Tuple[str, str]:
    """Require that path be of the form <module>:<qualname>."""
    module, qualname = module_path(path)
    if qualname is None:
        raise argparse.ArgumentTypeError('must be of the form <module>:<qualname>')
    return module, qualname


def monkeytype_config(path: str) -> Config:
    """Imports the config instance specified by path.

    Path should be in the form module:qualname. Optionally, path may end with (),
    in which case we will call/instantiate the given class/function.
    """
    should_call = False
    if path.endswith('()'):
        should_call = True
        path = path[:-2]
    module, qualname = module_path_with_qualname(path)
    try:
        config = get_name_in_module(module, qualname)
    except MonkeyTypeError as mte:
        raise argparse.ArgumentTypeError(f'cannot import {path}: {mte}')
    if should_call:
        config = config()
    return config


def display_sample_count(traces: List[CallTrace], stderr: IO) -> None:
    """Print to stderr the number of traces each stub is based on."""
    sample_counter = collections.Counter([t.funcname for t in traces])
    for name, count in sample_counter.items():
        print(f"Annotation for {name} based on {count} call trace(s).", file=stderr)


def get_stub(args: argparse.Namespace, stdout: IO, stderr: IO) -> Optional[Stub]:
    module, qualname = args.module_path
    thunks = args.config.trace_store().filter(module, qualname, args.limit)
    traces = []
    for thunk in thunks:
        try:
            traces.append(thunk.to_trace())
        except MonkeyTypeError as mte:
            print(f'ERROR: Failed decoding trace: {mte}', file=stderr)
    if not traces:
        return None
    rewriter = args.config.type_rewriter()
    if args.disable_type_rewriting:
        rewriter = NoOpRewriter()
    stubs = build_module_stubs_from_traces(
        traces,
        include_unparsable_defaults=args.include_unparsable_defaults,
        ignore_existing_annotations=args.ignore_existing_annotations,
        rewriter=rewriter,
    )
    if args.sample_count:
        display_sample_count(traces, stderr)
    return stubs.get(module, None)


def generate_stub_for_module(config: PostgresConfig,
                             module: str,
                             suppress_errors=False,
                             stderr=None,
                             limit_rows=300,
                             suppressed_exceptions=None) -> Optional[Stub]:
    suppressed_exceptions = (suppressed_exceptions or []) + [MonkeyTypeError]
    call_thunks = config.trace_store().filter(module)[:limit_rows]
    traces = []
    for thunk in call_thunks:
        try:
            try:
                if (thunk.qualname or '').startswith('_'):
                    continue

                trace = thunk.to_trace()
                traces.append(trace)
            except tuple(suppressed_exceptions):
                if not suppress_errors:
                    raise

        except MonkeyTypeError as mte:
            print(f'ERROR: Failed decoding trace: {mte}', file=stderr or sys.stderr)

    class_traces = []
    class_thunks = config.trace_store().extract_class_props(module)[:limit_rows]
    for thunk in class_thunks:
        try:
            class_trace = thunk.to_trace()
        except Exception:
            continue

        class_traces.append(class_trace)

    rewriter = config.type_rewriter()
    stubs = build_module_stubs_from_traces(
        traces,
        class_traces,
        ignore_existing_annotations=True,
        rewriter=rewriter
    )

    return stubs.get(module, None)


class HandlerError(Exception):
    pass


def run_handler(args: argparse.Namespace, stdout: IO, stderr: IO) -> None:
    # remove initial `monkeytype run`
    old_argv = sys.argv.copy()
    try:
        with trace(args.config):
            sys.argv = [args.script_path] + args.script_args
            if args.m:
                runpy.run_module(args.script_path, run_name='__main__', alter_sys=True)
            else:
                runpy.run_path(args.script_path, run_name='__main__')
    finally:
        sys.argv = old_argv


def update_args_from_config(args: argparse.Namespace) -> None:
    """Pull values from config for unspecified arguments."""
    if args.limit is None:
        args.limit = args.config.query_limit()
    if args.include_unparsable_defaults is None:
        args.include_unparsable_defaults = args.config.include_unparsable_defaults()


def main(argv: List[str], stdout: IO, stderr: IO) -> int:
    parser = argparse.ArgumentParser(
        description='Generate and apply stub files from collected type information.',
    )
    parser.add_argument(
        '--disable-type-rewriting',
        action='store_true', default=False,
        help="Show types without rewrite rules applied (default: False)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--include-unparsable-defaults',
        action='store_true', default=None,
        help=(
            "Include functions whose default values aren't valid Python expressions"
            " (default: False, unless changed in your config)"
        ),
    )
    group.add_argument(
        '--exclude-unparsable-defaults',
        action='store_false', default=None, dest='include_unparsable_defaults',
        help=(
            "Exclude functions whose default values aren't valid Python expressions"
            " (default: True, unless changed in your config)"
        ),
    )
    parser.add_argument(
        '--limit', '-l',
        type=int, default=None,
        help=(
            "How many traces to return from storage"
            " (default: 2000, unless changed in your config)"
        ),
    )
    parser.add_argument(
        '--config', '-c',
        type=monkeytype_config,
        default='monkeytype.config:get_default_config()',
        help=(
            "The <module>:<qualname> of the config to use"
            " (default: monkeytype_config:CONFIG if it exists, "
            "else monkeytype.config:DefaultConfig())"
        ),
    )

    subparsers = parser.add_subparsers(title="commands", dest="command")

    run_parser = subparsers.add_parser(
        'run',
        help='Run a Python script under MonkeyType tracing',
        description='Run a Python script under MonkeyType tracing')
    run_parser.add_argument(
        'script_path',
        type=str,
        help="""Filesystem path to a Python script file to run under tracing""")
    run_parser.add_argument(
        '-m',
        action='store_true',
        help="Run a library module as a script"
    )
    run_parser.add_argument(
        'script_args',
        nargs=argparse.REMAINDER,
    )
    run_parser.set_defaults(handler=run_handler)

    apply_parser = subparsers.add_parser(
        'apply',
        help='Generate and apply a stub',
        description='Generate and apply a stub')
    apply_parser.add_argument(
        'module_path',
        type=module_path,
        help="""A string of the form <module>[:<qualname>] (e.g.
my.module:Class.method). This specifies the set of functions/methods for which
we want to generate stubs.  For example, 'foo.bar' will generate stubs for
anything in the module 'foo.bar', while 'foo.bar:Baz' will only generate stubs
for methods attached to the class 'Baz' in module 'foo.bar'. See
https://www.python.org/dev/peps/pep-3155/ for a detailed description of the
qualname format.""")
    apply_parser.add_argument(
        "--sample-count",
        action='store_true',
        default=False,
        help='Print to stderr the numbers of traces stubs are based on'
    )
    apply_parser.set_defaults(handler=apply_stub_handler)

    stub_parser = subparsers.add_parser(
        'stub',
        help='Generate a stub',
        description='Generate a stub')
    stub_parser.add_argument(
        'module_path',
        type=module_path,
        help="""A string of the form <module>[:<qualname>] (e.g.
my.module:Class.method). This specifies the set of functions/methods for which
we want to generate stubs.  For example, 'foo.bar' will generate stubs for
anything in the module 'foo.bar', while 'foo.bar:Baz' will only generate stubs
for methods attached to the class 'Baz' in module 'foo.bar'. See
https://www.python.org/dev/peps/pep-3155/ for a detailed description of the
qualname format.""")
    stub_parser.add_argument(
        "--sample-count",
        action='store_true',
        default=False,
        help='Print to stderr the numbers of traces stubs are based on'
    )
    stub_parser.add_argument(
        "--ignore-existing-annotations",
        action='store_true',
        default=False,
        help='Ignore existing annotations and generate stubs only from traces.',
    )
    stub_parser.add_argument(
        "--diff",
        action='store_true',
        default=False,
        help='Compare stubs generated with and without considering existing annotations.',
    )
    stub_parser.set_defaults(handler=print_stub_handler)

    list_modules_parser = subparsers.add_parser(
        'list-modules',
        help='Listing of the unique set of module traces',
        description='Listing of the unique set of module traces')
    list_modules_parser.set_defaults(handler=list_modules_handler)

    args = parser.parse_args(argv)
    update_args_from_config(args)

    handler = getattr(args, 'handler', None)
    if handler is None:
        parser.print_help(file=stderr)
        return 1

    with args.config.cli_context(args.command):
        try:
            handler(args, stdout, stderr)
        except HandlerError as err:
            print(f"ERROR: {err}", file=stderr)
            return 1

    return 0


def entry_point_main() -> 'NoReturn':
    """Wrapper for main() for setuptools console_script entry point."""
    # Since monkeytype needs to import the user's code (and possibly config
    # code), the user's code must be on the Python path. But when running the
    # CLI script, it won't be. So we add the current working directory to the
    # Python path ourselves.
    sys.path.insert(0, os.getcwd())
    sys.exit(main(sys.argv[1:], sys.stdout, sys.stderr))
