import importlib
import importlib
import inspect
import pathlib
import shutil
import subprocess
from contextlib import contextmanager

from mypy import stubgen
from path import Path

from monkeytype import Config
from monkeytype.cli import generate_stub_for_module
from monkeytype.postgres.config import PostgresConfig
from monkeytype.rewriters.transformers import simplify_types, simplify_int_float, FindAcceptableCommonBase, \
    remove_empty_container, DepthFirstTypeTraverser, TwoElementUnionRewriter, SimplifyGenerics
from monkeytype.typing import ChainedRewriter, RewriteLargeUnion


ROOT_DIR = Path(__file__).parent.parent.parent
SOURCES_ROOT = ROOT_DIR / 'django_stubs_root'
STUBS_ROOT = SOURCES_ROOT / 'stubs' / 'django-stubs'
ROOT_DJANGO_DIR = SOURCES_ROOT / 'generator' / 'django_git'
TEXT_FILES_DIR = ROOT_DJANGO_DIR / 'tests'

with open(TEXT_FILES_DIR / 'test_apps.txt', 'r') as tests_dir_file:
    TEST_APPS_WITH_MODELS = tests_dir_file.read().strip().split('\n')

with open(TEXT_FILES_DIR / 'all_test_apps.txt', 'r') as all_tests_dir_file:
    ALL_TEST_APPS_SET = all_tests_dir_file.read().strip().split('\n')


class DjangoStubsConfig(PostgresConfig):
    @contextmanager
    def cli_context(self, command: str):
        import os
        import django

        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_sqlite')

        django.setup()
        yield

    def sample_rate(self):
        return 2

    def type_rewriter(self):
        two_element_transformers = [simplify_types,
                                    simplify_int_float,
                                    FindAcceptableCommonBase(allowed_bases_prefixes=['django']),
                                    remove_empty_container]
        traverser = DepthFirstTypeTraverser(
            union_rewriter=TwoElementUnionRewriter(
                two_element_transformers=[
                    *two_element_transformers,
                    SimplifyGenerics(two_element_transformers=two_element_transformers)
                ]
            ))
        return ChainedRewriter(rewriters=[traverser,
                                          RewriteLargeUnion(max_union_len=5)])


CONFIG = DjangoStubsConfig(relevant_modules=['django'],
                           connection_data={
                               'user': 'postgres',
                               'password': 'postgres',
                               'dbname': 'traces',
                               'host': '0.0.0.0'
                           })


def is_package(module: str) -> str:
    module = importlib.import_module(module)
    fpath = inspect.getfile(module)
    return fpath.endswith('__init__.py')


def get_path_to_stubfile(module, output_dir):
    path_to_stub_file = Path(output_dir)
    for module_part in module.split('.'):
        path_to_stub_file /= module_part
    if is_package(module):
        path_to_stub_file /= '__init__.pyi'
    else:
        path_to_stub_file += '.pyi'

    return path_to_stub_file


def generate_stub(module: str, config: Config, output_dir: str, line_length=80, suppress_errors=False):
    stubgen.generate_stub_for_module(module, output_dir=output_dir)

    path_to_stub_file = pathlib.Path(get_path_to_stubfile(module, output_dir=output_dir))
    shutil.move(path_to_stub_file, path_to_stub_file.with_suffix('.py'))

    stub_content = None
    stub = generate_stub_for_module(config, module, suppress_errors=suppress_errors)
    if stub is not None:
        stub_content = stub.render()

    if stub_content:
        stub_content = stub_content.replace('Ellipsis', '...')

        with open(path_to_stub_file, 'w') as stub_file:
            stub_file.write(stub_content)
    else:
        print('Stub file content is empty')

    pyi_dir = str(path_to_stub_file.parent)
    subprocess.run(['retype', '--pyi-dir', pyi_dir, '--target-dir', pyi_dir,
                    path_to_stub_file.with_suffix('.py')])

    shutil.move(path_to_stub_file.with_suffix('.py'), path_to_stub_file)

    subprocess.run(['black', '--line-length', str(line_length), '--pyi', path_to_stub_file])
    subprocess.run(['isort', '-a', 'from typing import Optional', path_to_stub_file])

    return path_to_stub_file