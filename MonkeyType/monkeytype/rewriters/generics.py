import inspect
from abc import ABCMeta, abstractmethod
from typing import Union, List, Dict, Set, Optional, Tuple, Callable, _Any, Any, Type, _Union

from monkeytype.typing import TypeRewriter


class UnionRewriter(TypeRewriter, metaclass=ABCMeta):
    @abstractmethod
    def transform_objects(self, obj1, obj2) -> Optional[Any]:
        pass

    def _traverse_all_pairs(self, collection, return_transformed_obj: Callable) -> Optional[Tuple[int, int, type]]:
        for i in range(len(collection)):
            left_op_elem = collection[i]
            for _inner, right_op_elem in enumerate(collection[i + 1:]):
                j = i + 1 + _inner
                transformed_obj = return_transformed_obj(left_op_elem, right_op_elem)
                if transformed_obj is not None:
                    return i, j, transformed_obj

        return None

    def rewrite_Union(self, union: Union):
        arg_classes = list(union.__args__)
        while True:
            if len(arg_classes) <= 1:
                break

            to_simplify = self._traverse_all_pairs(arg_classes, self.transform_objects)
            if to_simplify is None:
                break

            left_ind, right_ind, transformed_obj = to_simplify
            assert left_ind != right_ind

            arg_classes[left_ind] = transformed_obj
            del arg_classes[right_ind]

        return make_union(arg_classes)

    def rewrite_Union_one_operation(self, union: Union):
        arg_classes = list(union.__args__)
        if len(arg_classes) <= 1:
            return union

        to_simplify = self._traverse_all_pairs(arg_classes, self.transform_objects)
        if to_simplify is None:
            return union

        left_ind, right_ind, transformed_obj = to_simplify
        assert left_ind != right_ind

        arg_classes[left_ind] = transformed_obj
        del arg_classes[right_ind]

        return make_union(arg_classes)



# class FilterOutForeignClasses(TypeRewriter):
#     def rewrite(self, typ: type):
#         pass



#
# def _simplify_empty_container(obj1, obj2):
#     if not is_any_generic(obj1) or not is_any_generic(obj2):
#         return None
#
#     if not type(obj1) == type(obj2):
#         return None
#
#     computed_args = []
#     for arg_class in obj1.__args__:
#         if not is_any(arg_class):
#             break
#     else:
#         computed_args.append(obj1)


class RemoveEmptyContainersRewriter(UnionRewriter):
    def _is_empty(self, typ):
        args = getattr(typ, '__args__', [])
        return args and all(isinstance(e, _Any) for e in args)

    def transform_objects(self, obj1, obj2):
        if not is_any_generic(obj1) or not is_any_generic(obj2):
            return None

        if not obj1._gorg == obj2._gorg:
            return None

        if self._is_empty(obj1):
            return obj2

        if self._is_empty(obj2):
            return obj1


ALLOWED_BASE_CLASS_MODULES = ['rewriters', 'test_generics']


def is_allowed_base_class(base_cls) -> bool:
    for allowed_module in ALLOWED_BASE_CLASS_MODULES:
        if base_cls.__module__.startswith(allowed_module + '.') or base_cls.__module__ == allowed_module:
            return True
    return False


def _find_acceptable_common_base(cls1, cls2) -> Optional[type]:
    if cls1 == cls2:
        return cls1

    for base1 in inspect.getmro(cls1):
        if is_allowed_base_class(base1):
            for base2 in inspect.getmro(cls2):
                if base1 == base2:
                    return base1

    return None


# def _simplify_lists(lst1, lst2):
#     base = _find_acceptable_common_base(lst1.__args__[0], lst2.__args__[0])
#     if base is not None:
#         return List[base]


# def _simplify_sets(set1, set2):
#     base = _find_acceptable_common_base(set1.__args__[0], set2.__args__[0])
#     if base is not None:
#         return Set[base]


# def _simplify_dicts(dict1, dict2):
#     new_key_base = _find_acceptable_common_base(dict1.__args__[0], dict2.__args__[0])
#     new_value_base = _find_acceptable_common_base(dict1.__args__[1], dict2.__args__[1])
#
#     if new_key_base and new_value_base:
#         return Dict[new_key_base, new_value_base]

class MainRewriter(TypeRewriter):
    def __init__(self, *,
                 two_element_union_transformers: List[Callable] = None,
                 one_element_transformers: List[Callable] = None,
                 information_losing_one_element_transformers: List[Callable] = None):
        self.two_element_union_transformers = two_element_union_transformers or []
        self.one_element_transformers = one_element_transformers or []
        self.information_losing_one_element_transformers = information_losing_one_element_transformers or []

    def rewrite_Union(self, union: _Union):
        union = super().rewrite_Union(union)
        while True:
            simplified = False

            for transformer in self.two_element_union_transformers:
                new_union = transformer(union)
                is_equal = (union == new_union)
                union = new_union

                if not is_equal:
                    simplified = True
                    break

            if not simplified:
                break

        return union

    def generic_rewrite(self, typ: type):
        for transformer in self.one_element_transformers:
            typ = transformer(typ)

        for transformer in self.information_losing_one_element_transformers:
            typ = transformer(typ)

        return typ





class SimplifyByMROTraversing(UnionRewriter):
    def transform_objects(self, obj1, obj2):
        if type(obj1) != type(obj2):
            return None

        if not hasattr(obj1, '_gorg'):
            return _find_acceptable_common_base(obj1, obj2)

        transformed_args = []
        for obj1_arg, obj2_arg in zip(obj1.__args__, obj2.__args__):
            common_base = _find_acceptable_common_base(obj1_arg, obj2_arg)
            if common_base is None:
                return None
            transformed_args.append(common_base)

        return obj1._gorg[tuple(transformed_args)]


class SimplifyUnionOfTuples(TypeRewriter):
    def rewrite_Union(self, union: Union):
        return union


class SimplifyIntFloat(TypeRewriter):
    def rewrite_Union(self, union):
        if float in union.__args__ and int in union.__args__:
            new_args = list(union.__args__)
            new_args.remove(int)
            return make_union(new_args)

        return union


class SimplifyUnionOfTypes(UnionRewriter):
    def transform_objects(self, obj1, obj2):
        if not is_generic(obj1, Type) or not is_generic(obj2, Type):
            return None

        return Type[make_union(list(obj1.__args__) + list(obj2.__args__))]


class DepthFirstTypeTraverser(TypeRewriter):
    def __init__(self, union_rewriter: TypeRewriter):
        self.union_rewriter = union_rewriter

    def rewrite(self, typ: type):
        arg_classes = getattr(typ, '__args__', None)
        if not arg_classes:
            return typ

        if hasattr(typ, '_gorg'):
            processed_args = []
            for arg_class in arg_classes:
                processed_args.append(self.rewrite(arg_class))
            return typ._gorg[tuple(processed_args)]

        processed_args = []
        for arg_class in arg_classes:
            processed_args.append(self.rewrite(arg_class))

        new_union = make_union(processed_args)
        return self.union_rewriter.rewrite_Union(new_union)