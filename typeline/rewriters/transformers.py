import datetime
import inspect
import logging
from pathlib import Path
from typing import Union, _Any, _Union, List, Set, Dict, Tuple, Optional, Type, Callable, Any

import typing_inspect

from typeline.config import LIB_PATHS
from typeline.typing import TypeRewriter

logger = logging.getLogger('root')
logger.setLevel(logging.INFO)


class NotDefined(object):
    pass


NOT_SIMPLIFIED = NotDefined()


def is_generic(cls, generic) -> bool:
    return hasattr(cls, '_gorg') and cls._gorg == generic


def is_any(cls):
    return issubclass(cls, _Any)


def make_union(args):
    return Union[tuple(args)]


def _is_empty(typ):
    args = getattr(typ, '__args__', [])
    return args and all(isinstance(e, _Any) for e in args)


def remove_empty_container(obj1, obj2):
    if not typing_inspect.is_generic_type(obj1) or not typing_inspect.is_generic_type(obj2):
        return NOT_SIMPLIFIED

    if obj1._gorg != obj2._gorg:
        return NOT_SIMPLIFIED

    if _is_empty(obj1):
        return obj2

    if _is_empty(obj2):
        return obj1

    return NOT_SIMPLIFIED


ALLOWED_BASE_CLASS_MODULES = ['rewriters', 'test_generics']


def _is_allowed_base_class(base_cls, allowed_bases=None) -> bool:
    for allowed_module in allowed_bases or ALLOWED_BASE_CLASS_MODULES:
        if base_cls.__module__.startswith(allowed_module + '.') or base_cls.__module__ == allowed_module:
            return True
    return False


class FindAcceptableCommonBase(object):
    def __init__(self, allowed_bases_prefixes, allowed_bases=None):
        self.allowed_bases_prefixes = allowed_bases_prefixes
        self.allowed_bases = allowed_bases

    def __call__(self, obj1, obj2) -> type:
        return find_acceptable_common_base(obj1, obj2, self.allowed_bases_prefixes,
                                           allowed_base_classes=self.allowed_bases)


def find_acceptable_common_base(cls1, cls2, allowed_bases=None, allowed_base_classes=None) -> Optional[type]:
    if cls1 == cls2:
        return cls1

    if isinstance(cls1, (_Any, _Union)) or hasattr(cls1, '_gorg'):
        return NOT_SIMPLIFIED

    if isinstance(cls2, (_Any, _Union)) or hasattr(cls2, '_gorg'):
        return NOT_SIMPLIFIED

    allowed_base_classes = allowed_base_classes or set()
    for base1 in inspect.getmro(cls1):
        if _is_allowed_base_class(base1, allowed_bases=allowed_bases) or base1 in allowed_base_classes:
            for base2 in inspect.getmro(cls2):
                if base1 == base2:
                    return base1

    return NOT_SIMPLIFIED


def find_acceptable_common_abc(cls1, cls2):
    if cls1 == cls2:
        return cls1

    if isinstance(cls1, (_Any, _Union)) or isinstance(cls2, (_Any, _Union)):
        return NOT_SIMPLIFIED

    for base1 in inspect.getmro(cls1):
        if base1.__module__ == 'collections.abc':
            for base2 in inspect.getmro(cls2):
                print(f'looking for base for abc for {base1.__name__}:', base2)
                if base2.__module__ == '_collections_abc' or base2.__module__ == 'typing' \
                        and base2.__name__ == base1.__name__:
                    return base2

    for base1 in inspect.getmro(cls2):
        if base1.__module__ == 'collections.abc':
            for base2 in inspect.getmro(cls1):
                print(f'looking for base for abc for {base1.__name__}:', base2)
                if base2.__module__ == '_collections_abc' or base2.__module__ == 'typing' \
                        and base2.__name__ == base1.__name__:
                    return base2

    return NOT_SIMPLIFIED


def simplify_int_float(obj1, obj2):
    if {obj1, obj2} == {int, float}:
        return float

    return NOT_SIMPLIFIED


def simplify_datetime_classes(obj1, obj2):
    if obj1.__module__ == 'datetime' and obj2.__module__ == 'datetime':
        return datetime.datetime

    return NOT_SIMPLIFIED


class SimplifyTypes(object):
    def __init__(self, union_rewriter):
        self.union_rewriter = union_rewriter

    def __call__(self, obj1, obj2):
        result = simplify_types(obj1, obj2, self.union_rewriter)
        return result


def simplify_types(obj1, obj2, union_rewriter):
    if not is_generic(obj1, Type) or not is_generic(obj2, Type):
        return NOT_SIMPLIFIED

    union = make_union(list(obj1.__args__) + list(obj2.__args__))
    rewritten_union = union_rewriter.rewrite_Union(union)
    if rewritten_union is not NOT_SIMPLIFIED:
        union = rewritten_union

    return Type[union]


class CollapseUnions(object):
    def __init__(self, union_rewriter):
        self.union_rewriter = union_rewriter
        self.increment = 0

    def __call__(self, obj1, obj2):
        if self.increment > 5:
            return NOT_SIMPLIFIED

        self.increment += 1
        result = collapse_unions(obj1, obj2, self.union_rewriter)
        self.increment -= 1
        return result


class IdentityUnionRewriter(object):
    def rewrite(self, union):
        return union


def collapse_unions(obj1, obj2,
                    union_rewriter=None):
    if not union_rewriter:
        union_rewriter = IdentityUnionRewriter()

    union = make_union([obj1, obj2])
    union = union_rewriter.rewrite(union)
    if not typing_inspect.is_union_type(union):
        return union

    for obj in [obj1, obj2]:
        if typing_inspect.is_union_type(obj) and len(union.__args__) <= len(obj.__args__):
            return union

    return NOT_SIMPLIFIED


class SimplifyGenerics(object):
    def __init__(self, two_element_transformers: List[Callable]):
        self.two_element_transformers = two_element_transformers

    def __call__(self, obj1, obj2):
        return simplify_generics(obj1, obj2, self.two_element_transformers)


def simplify_generics(obj1, obj2, two_element_transformers: List[Callable]):
    if not typing_inspect.is_generic_type(obj1) or not typing_inspect.is_generic_type(obj2):
        return NOT_SIMPLIFIED

    if typing_inspect.get_origin(obj1) != typing_inspect.get_origin(obj2):
        return NOT_SIMPLIFIED

    # print()
    # print('trying to simplify', obj1, obj2, sep='\n')

    transformed = []
    for arg1, arg2 in zip(obj1.__args__, obj2.__args__):
        # print('arg1, arg2: ', arg1, arg2)

        for transformer in [collapse_unions, *two_element_transformers]:
            transformed_obj = transformer(arg1, arg2)
            if transformed_obj is not NOT_SIMPLIFIED:
                transformed.append(transformed_obj)
                break
        else:
            return NOT_SIMPLIFIED

    result = typing_inspect.get_origin(obj1)[tuple(transformed)]

    # print('result', result)
    return result
    # for transformer in two_element_transformers:
    #     transformed = []
    #     for obj1_arg, obj2_arg in zip(obj1.__args__, obj2.__args__):
    #         transformed_obj = collapse_unions(obj1_arg, obj2_arg)
    #         if transformed_obj is not None:
    #             # elements are equal with respect to union, could simplify
    #             transformed.append(transformed_obj)
    #             continue
    #
    #         transformed_obj = transformer(obj1_arg, obj2_arg)
    #         if transformed_obj is not None:
    #             # elements are equal with respect to the transformer
    #             transformed.append(transformed_obj)
    #             continue
    #
    #         break
    #     else:
    #         return obj1._gorg[tuple(transformed)]

    # return None


class SimplifyTuples(TypeRewriter):
    def __init__(self, two_element_transformers: List[Callable]):
        self.two_element_transformers = two_element_transformers

    def __call__(self, obj1, obj2):
        return simplify_tuples(obj1, obj2,
                               [*self.two_element_transformers])


def simplify_tuples(tup1, tup2, two_element_transformers):
    for t in [tup1, tup2]:
        if not typing_inspect.is_tuple_type(t):
            return NOT_SIMPLIFIED

        args = t.__args__
        if not args:
            return NOT_SIMPLIFIED

    transformed = []
    merge_used = False
    for i, (obj1_arg, obj2_arg) in enumerate(zip(tup1.__args__,
                                                 tup2.__args__)):
        simplified = False
        for transformer in [collapse_unions, *two_element_transformers]:
            transformed_obj = transformer(obj1_arg, obj2_arg)
            if transformed_obj is not NOT_SIMPLIFIED:
                transformed.append(transformed_obj)
                simplified = True
                break

        if not simplified:
            if not merge_used:
                transformed.append(make_union([obj1_arg, obj2_arg]))
                merge_used = True
            else:
                return NOT_SIMPLIFIED

    return Tuple[tuple(transformed)]


class TwoElementUnionRewriter(TypeRewriter):
    def __init__(self, two_element_transformers: List[Callable]):
        self.two_element_transformers = two_element_transformers

    def _traverse_all_pairs(self, collection, transformers: List[Callable]) -> Optional[Tuple[int, int, type]]:
        for i in range(len(collection)):
            left_op_elem = collection[i]
            for _inner, right_op_elem in enumerate(collection[i + 1:]):
                j = i + 1 + _inner

                for transformer in transformers:
                    transformed_obj = transformer(left_op_elem, right_op_elem)
                    if transformed_obj is not NOT_SIMPLIFIED:
                        return i, j, transformed_obj

        return NOT_SIMPLIFIED

    def rewrite_Union(self, union: Union):
        if not typing_inspect.is_union_type(union):
            return union

        arg_classes = list(union.__args__)
        while True:
            if len(arg_classes) <= 1:
                break

            to_simplify = self._traverse_all_pairs(arg_classes, self.two_element_transformers)
            if to_simplify is NOT_SIMPLIFIED:
                break

            left_ind, right_ind, transformed_obj = to_simplify
            assert left_ind != right_ind

            arg_classes[left_ind] = transformed_obj
            del arg_classes[right_ind]

        return make_union(arg_classes)


# def with_black(obj):
#     process = subprocess.Popen(['black', '--quiet', '--fast', '-'], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
#     process.stdin.write(str(obj).encode())
#     stdout = process.communicate()[0].decode()
#     return stdout


# def print_with_black(message, typ):
#     print(message, end=' ')
#     print(black.format_str(str(typ), line_length=80))


class DepthFirstTypeTraverser(TypeRewriter):
    def __init__(self, union_rewriter: TwoElementUnionRewriter, num_of_passes=1):
        self.union_rewriter = union_rewriter
        self.num_of_passes = num_of_passes

    def _rewrite(self, typ: type):
        if isinstance(typ, _Any):
            return typ

        arg_classes = getattr(typ, '__args__', [])
        if not arg_classes:
            return typ

        if typing_inspect.is_generic_type(typ) or typing_inspect.is_tuple_type(typ):
            gorg = typing_inspect.get_origin(typ)

            processed_args = []
            for arg_class in arg_classes:
                processed = self._rewrite(arg_class)
                if processed is NOT_SIMPLIFIED:
                    processed = arg_class

                processed_args.append(processed)

            return gorg[tuple(processed_args)]

        processed_args = []
        for arg_class in arg_classes:
            rewritten = self._rewrite(arg_class)
            if rewritten is NOT_SIMPLIFIED:
                processed_args.append(arg_class)
                continue

            processed_args.append(rewritten)

        new_union = make_union(processed_args)
        new_union = self.union_rewriter.rewrite_Union(new_union)

        return new_union

    def rewrite(self, typ: type):
        rewritten = typ
        for i in range(self.num_of_passes):
            rewritten = self._rewrite(typ)

        return rewritten


def simplify_to_abstract_class(obj1, obj2):
    if hasattr(obj1, '_gorg') or isinstance(obj1, _Union) or isinstance(obj1, _Any):
        return NOT_SIMPLIFIED

    if not hasattr(obj2, '_gorg'):
        return NOT_SIMPLIFIED

    for base in inspect.getmro(obj1):
        if base.__name__ == 'dict' and obj2._gorg == Dict:
            return obj2

        if base.__name__ == 'list' and obj2._gorg == List:
            return obj2

    return NOT_SIMPLIFIED

    # for base in inspect.getmro(obj1):
    #     if base.__module__ == 'collections.abc':
    #         for obj2_base in inspect.getmro(obj2):
    #             abc = getattr(importlib.import_module('typing'), obj2_base.__name__, None)
    #             if abc is not None and base.__name__ == abc.__name__:
    #                 return abc[tuple(obj2.__args__)]


def is_from_stdlib(typ: type):
    if typ.__module__ == 'builtins':
        return True

    try:
        filename = Path(inspect.getsourcefile(typ))
    except TypeError:
        return True

    for lib_path in LIB_PATHS:
        try:
            filename.relative_to(lib_path)
            return True
        except ValueError:
            pass
    return False


class MroSimplifier(TypeRewriter):
    def __init__(self, allowed_modules, include_stdlib=True):
        self.allowed_modules = allowed_modules
        self.include_stdlib = include_stdlib

    def _simplify_by_mro(self, typ: type):
        for base in inspect.getmro(typ)[:-1]:
            if self.include_stdlib and is_from_stdlib(base):
                return base

            for allowed_module in self.allowed_modules:
                if base.__module__.startswith(allowed_module):
                    return base

        return Any

    def rewrite(self, typ: type):
        if isinstance(typ, _Any):
            return typ

        arg_classes = getattr(typ, '__args__', None)
        if not arg_classes:
            return self._simplify_by_mro(typ)

        if hasattr(typ, '_gorg'):
            processed_args = []
            for arg_class in arg_classes:
                processed_args.append(self.rewrite(arg_class))

            simplified = typ._gorg[tuple(processed_args)]
            return simplified

        processed_args = []
        for arg_class in arg_classes:
            rewritten = self.rewrite(arg_class)
            if not isinstance(rewritten, _Any):
                processed_args.append(rewritten)

        if not processed_args:
            return Any

        new_union = make_union(processed_args)
        return new_union


class TupleSimplifier(TypeRewriter):
    def __init__(self, max_members=10):
        self.max_members = max_members

    def _simplify_big_tuple(self, typ: Tuple):
        if not typing_inspect.is_tuple_type(typ):
            return typ

        arg_classes = typ.__args__
        if not arg_classes:
            return typ

        if len(arg_classes) > self.max_members and len(set(arg_classes)) == 1:
            return Tuple[arg_classes[0], ...]

        if len(arg_classes) > self.max_members:
            return Tuple[Any, ...]

        return typ

    def rewrite(self, typ: type):
        if isinstance(typ, _Any):
            return typ

        arg_classes = getattr(typ, '__args__', None)
        if not arg_classes:
            return typ

        if typing_inspect.is_tuple_type(typ):
            processed_args = []
            for arg_class in arg_classes:
                processed_args.append(self.rewrite(arg_class))
            tup = Tuple[tuple(processed_args)]
            return self._simplify_big_tuple(tup)

        if hasattr(typ, '_gorg'):
            processed_args = []
            for arg_class in arg_classes:
                processed_args.append(self.rewrite(arg_class))

            simplified = typ._gorg[tuple(processed_args)]
            return simplified

        processed_args = []
        for arg_class in arg_classes:
            rewritten = self.rewrite(arg_class)
            if not isinstance(rewritten, _Any):
                processed_args.append(rewritten)

        if not processed_args:
            return Any

        new_union = make_union(processed_args)
        return new_union
#
# class TuplesRewriter(TypeRewriter):
#     def rewrite(self, typ: type):
#         pass
