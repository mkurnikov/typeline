import inspect
import logging
from typing import Union, _Any, _Union, List, Set, Dict, Tuple, Optional, Type, Callable, Any

from monkeytype.typing import TypeRewriter

logger = logging.getLogger('root')
logger.setLevel(logging.INFO)


def is_any_generic(cls):
    for base in [List, Set, Dict, Tuple]:
        if is_generic(cls, base):
            return True

    return False


def is_generic(cls, generic) -> bool:
    return hasattr(cls, '_gorg') and cls._gorg == generic


def safe_is_subclass(cls, base):
    try:
        issubclass(cls, base)
    except TypeError:
        return False


def is_parametrized_generic(cls, base):
    pass


def is_any(cls):
    return issubclass(cls, _Any)


def is_union(cls):
    return isinstance(cls, _Union)


def make_union(args):
    return Union[tuple(args)]


def _is_empty(typ):
    args = getattr(typ, '__args__', [])
    return args and all(isinstance(e, _Any) for e in args)


def remove_empty_container(obj1, obj2):
    if not is_any_generic(obj1) or not is_any_generic(obj2):
        return None

    if not obj1._gorg == obj2._gorg:
        return None

    if _is_empty(obj1):
        return obj2

    if _is_empty(obj2):
        return obj1


ALLOWED_BASE_CLASS_MODULES = ['rewriters', 'test_generics']


def _is_allowed_base_class(base_cls, allowed_bases=None) -> bool:
    for allowed_module in allowed_bases or ALLOWED_BASE_CLASS_MODULES:
        if base_cls.__module__.startswith(allowed_module + '.') or base_cls.__module__ == allowed_module:
            return True
    return False


class FindAcceptableCommonBase(object):
    def __init__(self, allowed_bases_prefixes):
        self.allowed_bases_prefixes = allowed_bases_prefixes

    def __call__(self, obj1, obj2) -> type:
        return find_acceptable_common_base(obj1, obj2, self.allowed_bases_prefixes)


def find_acceptable_common_base(cls1, cls2, allowed_bases=None) -> Optional[type]:
    if cls1 == cls2:
        return cls1

    if isinstance(cls1, (_Any, _Union)) or hasattr(cls1, '_gorg'):
        return None

    if isinstance(cls2, (_Any, _Union)) or hasattr(cls2, '_gorg'):
        return None

    for base1 in inspect.getmro(cls1):
        if _is_allowed_base_class(base1, allowed_bases=allowed_bases):
            for base2 in inspect.getmro(cls2):
                if base1 == base2:
                    return base1

    return None


def simplify_int_float(obj1, obj2):
    if {obj1, obj2} == {int, float}:
        return float

    return None


def simplify_types(obj1, obj2):
    if not is_generic(obj1, Type) or not is_generic(obj2, Type):
        return None

    return Type[make_union(list(obj1.__args__) + list(obj2.__args__))]


def _union_transformer(obj1, obj2):
    union = make_union([obj1, obj2])
    if not isinstance(union, _Union):
        return union

    for obj in [obj1, obj2]:
        if isinstance(obj, _Union) and len(union.__args__) <= len(obj.__args__):
            return union

    return None


class SimplifyGenerics(object):
    def __init__(self, two_element_transformers: List[Callable]):
        self.two_element_transformers = two_element_transformers

    def __call__(self, obj1, obj2):
        return simplify_generics(obj1, obj2,
                                 [*self.two_element_transformers, SimplifyGenerics(self.two_element_transformers)])


def simplify_generics(obj1, obj2, two_element_transformers: List[Callable]):
    if not hasattr(obj1, '_gorg') or not hasattr(obj2, '_gorg'):
        return None

    if obj1._gorg != obj2._gorg:
        return None

    for transformer in two_element_transformers:
        transformed = []
        for obj1_arg, obj2_arg in zip(obj1.__args__, obj2.__args__):
            transformed_obj = _union_transformer(obj1_arg, obj2_arg)
            if transformed_obj is not None:
                # elements are equal with respect to union, could simplify
                transformed.append(transformed_obj)
                continue

            transformed_obj = transformer(obj1_arg, obj2_arg)
            if transformed_obj is not None:
                # elements are equal with respect to the transformer
                transformed.append(transformed_obj)
                continue

            break
        else:
            return obj1._gorg[tuple(transformed)]

    return None


class SimplifyTuples(TypeRewriter):
    def __init__(self, two_element_transformers: List[Callable]):
        self.two_element_transformers = two_element_transformers

    def __call__(self, obj1, obj2):
        return simplify_tuples(obj1, obj2,
                               [*self.two_element_transformers])


def simplify_tuples(tup1, tup2, two_element_transformers):
    for t in [tup1, tup2]:
        if not hasattr(t, '_gorg') or t._gorg != Tuple:
            return None

    if len(tup1.__args__) != len(tup2.__args__):
        return None

    not_equal_index = -1
    transformed = []
    for i, (obj1_arg, obj2_arg) in enumerate(zip(tup1.__args__, tup2.__args__)):
        simplified = False
        for transformer in [_union_transformer, *two_element_transformers]:
            if simplified:
                break

            transformed_obj = transformer(obj1_arg, obj2_arg)
            if transformed_obj is not None:
                transformed.append(transformed_obj)
                simplified = True
                continue

        if not simplified:
            if not_equal_index != -1:
                return None
            else:
                not_equal_index = i

    transformed.insert(i, Union[tup1.__args__[i], tup2.__args__[i]])
    return Tuple[tuple(transformed)]

    #
    #
    #
    # transformed = []
    #
    # simplified = False
    # for transformer in two_element_transformers:
    #     transformed_obj = _union_transformer(obj1_arg, obj2_arg)
    #     if transformed_obj is not None:
    #         # elements are equal with respect to union, could simplify
    #         transformed.append(transformed_obj)
    #         continue
    #
    #     transformed_obj = transformer(obj1_arg, obj2_arg)
    #     if transformed_obj is not None:
    #         # elements are equal with respect to the transformer
    #         transformed.append(transformed_obj)
    #         continue
    #
    #     if not_equal_index != -1:
    #         return None
    #
    #     not_equal_index = i

    # obj_union = make_union([obj1_arg, obj2_arg])
    # # not equal (with respect to union operation)
    # if is_union(obj_union):
    #
    # transformed_obj = transformer(obj1_arg, obj2_arg)
    # if transformed_obj is None:
    #     break

    # transformed_args.append(common_base)

    # return obj1._gorg[tuple(transformed_args)]


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
                    if transformed_obj is not None:
                        return i, j, transformed_obj

        return None

    def rewrite_Union(self, union: Union):
        arg_classes = list(union.__args__)
        while True:
            if len(arg_classes) <= 1:
                break

            to_simplify = self._traverse_all_pairs(arg_classes, self.two_element_transformers)
            if to_simplify is None:
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

        arg_classes = getattr(typ, '__args__', None)
        if not arg_classes:
            return typ

        if hasattr(typ, '_gorg'):
            processed_args = []
            for arg_class in arg_classes:
                processed_args.append(self._rewrite(arg_class))
            simplified = typ._gorg[tuple(processed_args)]

            return simplified

        processed_args = []
        for arg_class in arg_classes:
            rewritten = self._rewrite(arg_class)
            if not isinstance(rewritten, _Any):
                processed_args.append(self._rewrite(arg_class))

        new_union = make_union(processed_args)
        new_union = self.union_rewriter.rewrite_Union(new_union)

        return new_union

    def rewrite(self, typ: type):
        rewritten = typ
        for i in range(self.num_of_passes):
            rewritten = self._rewrite(typ)

        return rewritten


class MroSimplifier(TypeRewriter):
    def __init__(self, filtered_modules):
        self.filtered_modules = filtered_modules

    def _simplify_by_mro(self, typ):
        for base in inspect.getmro(typ)[:-1]:
            filtered = False
            for filtered_module in self.filtered_modules:
                if base.__module__.startswith(filtered_module):
                    filtered = True
                    break

            if not filtered:
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

#
# class TuplesRewriter(TypeRewriter):
#     def rewrite(self, typ: type):
#         pass
