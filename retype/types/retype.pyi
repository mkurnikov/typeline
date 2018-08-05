from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Type, Union

from lib2to3.pytree import Node, Leaf
from typed_ast import ast3

_LN = Union[Node, Leaf]
Callbacks = List[Callable[[], None]]

def new(n: _LN, prefix: str = None) -> _LN: ...
def fix_line_numbers(body: Node) -> None: ...
def name_used_in_node(node: _LN, name: Leaf) -> bool: ...
def get_offset_and_prefix(body: Node, skip_assignments: bool = False) -> Tuple[int, str]: ...
def get_annotated_param(node: _LN, arg: ast3.arg, *, missing_ok: bool = False) -> _LN: ...
def gen_annotated_params(args: Iterable[ast3.arg], defaults: Iterable[Optional[ast3.expr]], params: List[_LN], *, implicit_default: bool = False, is_method: bool = False) -> Iterator[_LN]: ...
def pop_param(params: List[_LN]) -> Tuple[_LN, Union[Node, Leaf, None]]: ...
def flatten_some(children: List[_LN]) -> Iterator[_LN]: ...
def maybe_space_before_comment(text: Optional[str]) -> str: ...
def minimize_whitespace(text: str) -> str: ...
def remove_function_signature_type_comment(body: _LN) -> None: ...
def maybe_replace_any_if_equal(name: str, expected: _LN, actual: _LN) -> _LN: ...
def ensure_annotations_equal(name: str, expected: _LN, actual: _LN) -> None: ...
def ensure_no_annotation(ann: Optional[ast3.AST]) -> None: ...
def copy_type_comment_to_annotation(arg: ast3.arg) -> None: ...
def copy_type_comments_to_annotations(args: ast3.arguments) -> None: ...
def copy_arguments_to_annotations(args: ast3.arguments, type_comment: Union[ast3.expr, List[ast3.expr]], *, is_method: bool = False) -> None: ...
def parse_arguments(arguments: str) -> ast3.arguments: ...
def parse_type_comment(type_comment: str) -> ast3.expr: ...
def parse_signature_type_comment(type_comment: str) -> Tuple[Union[ast3.expr, List[ast3.expr]], ast3.expr]:
    argtypes: Union[ast3.expr, List[ast3.expr]]

def get_function_signature(fun: ast3.FunctionDef, *, is_method: bool = False) -> Tuple[ast3.arguments, Optional[ast3.expr]]: ...
def annotate_return(function: List[_LN], ast_returns: Optional[ast3.AST], offset: int) -> None: ...
def annotate_parameters(parameters: _LN, ast_args: ast3.arguments, *, is_method: bool = False) -> None:
    typedargslist: List[_LN]
    defaults: List[Optional[ast3.expr]]

def append_after_imports(stmt_to_insert: Node, node: Node) -> None: ...
def make_import(*names: ast3.alias, from_module: str = None) -> Node:
    imports: List[_LN]
    imports_and_commas: List[_LN]
    result: List[_LN]

def is_builtin_method_decorator(name: str) -> bool: ...
def is_assignment(node: _LN) -> bool: ...
def fix_signature_annotation_type_comment(node: Node, last: Node, *, offset: int) -> None: ...
def fix_variable_annotation_type_comment(node: _LN, last: Node) -> None: ...
def fix_remaining_type_comments(node: Node) -> None: ...
def decorator_names(obj: Union[Node, ast3.AST, List[Union[Node, ast3.AST]]]) -> List[str]: ...
def names_already_imported(names: Union[List[ast3.AST], ast3.AST], node: Node) -> bool: ...
def convert_annotation(ann: ast3.AST) -> _LN: ...
def serialize_attribute(attr: ast3.AST) -> str: ...
def reapply(ast_node: ast3.AST, lib2to3_node: Node) -> Callbacks: ...
def reapply_all(ast_node: List[ast3.stmt], lib2to3_node: Node) -> None: ...
def lib2to3_parse(src_txt: str) -> Node: ...
def lib2to3_unparse(node: Node, *, hg: bool = False) -> str: ...
def retype_file(src: Path, pyi_dir: Path, targets: Path, *, quiet: bool = False, hg: bool = False) -> Path: ...
def retype_path(src: Path, pyi_dir: Path, targets: Path, *, src_explicitly_given: bool = False, quiet: bool = False, hg: bool = False) -> Iterator[Tuple[Path, str, Type[Exception], List[str]]]: ...
def main(src: Sequence[str], pyi_dir: str, target_dir: str, incremental: bool, quiet: bool, replace_any: bool, hg: bool, traceback: bool) -> None: ...

# internal singledispatch implementations, etc.
def _r_list(l: List[ast3.AST], lib2to3_node: Node) -> Callbacks: ...
def _r_importfrom(import_from: ast3.ImportFrom, node: Node) -> Callbacks: ...
def _r_import(import_: ast3.Import, node: Node) -> Callbacks: ...
def _r_classdef(cls: ast3.ClassDef, node: Node) -> Callbacks: ...
def _r_functiondef(fun: ast3.FunctionDef, node: Node) -> Callbacks: ...
def _r_annassign(annassign: ast3.AnnAssign, body: Node) -> Callbacks: ...
def _r_assign(assign: ast3.Assign, body: Node) -> Callbacks: ...

def _sa_attribute(attr: ast3.Attribute) -> str: ...
def _sa_name(name: ast3.Name) -> str: ...
def _sa_expr(expr: ast3.Expr) -> str: ...

def _c_subscript(sub: ast3.Subscript) -> Node: ...
def _c_name(name: ast3.Name) -> Leaf: ...
def _c_nameconstant(const: ast3.NameConstant) -> Leaf: ...
def _c_ellipsis(ell: ast3.Ellipsis) -> Node: ...
def _c_str(s: ast3.Str) -> Leaf: ...
def _c_index(index: ast3.Index) -> _LN: ...
def _c_tuple(tup: ast3.Tuple) -> Node: ...
def _c_attribute(attr: ast3.Attribute) -> Leaf: ...
def _c_call(call: ast3.Call) -> Node: ...
def _c_keyword(kwarg: ast3.keyword) -> Node: ...
def _c_list(l: ast3.List) -> Node: ...

def _nai_list(names: List[ast3.alias], node: Node) -> bool: ...
def _nai_alias(alias: ast3.alias, node: Node) -> bool: ...

def _dn_list(l: List[Union[Node, ast3.AST]]) -> List[str]: ...
def _dn_node(node: Node) -> List[str]: ...
def _dn_name(name: ast3.Name) -> List[str]: ...
def _dn_call(call: ast3.Call) -> List[str]: ...
def _dn_attribute(attr: ast3.Attribute) -> List[str]: ...

def _nuin_node(node: Node, name: Leaf) -> bool: ...
def _nuin_leaf(leaf: Leaf, name: Leaf) -> bool: ...