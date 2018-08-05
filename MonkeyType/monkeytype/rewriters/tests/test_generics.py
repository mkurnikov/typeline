from functools import partial
from typing import Union, Any, List, Dict, Set, Tuple, Type, Iterator
from unittest import TestCase

from django.http.multipartparser import LazyStream
from monkeytype.rewriters.transformers import DepthFirstTypeTraverser, TwoElementUnionRewriter, simplify_types, \
    simplify_int_float, find_acceptable_common_base, remove_empty_container, simplify_generics, SimplifyGenerics, \
    FindAcceptableCommonBase
from monkeytype.typing import TypeRewriter


class BaseRewriterTestCase(TestCase):
    rewriter: TypeRewriter

    def assertRewrittenEqual(self, original, goal):
        self.assertEqual(self.rewriter.rewrite(original), goal)


class TestRemoveEmptyContainers(BaseRewriterTestCase):
    def setUp(self):
        self.rewriter = RemoveEmptyContainersRewriter()

    def test_no_other_lists_leave_as_is(self):
        self.assertRewrittenEqual(Union[str, List[Any]], Union[str, List[Any]])
        self.assertRewrittenEqual(
            Union[Dict[str, Any], List[Any]], Union[Dict[str, Any], List[Any]]
        )

    def test_if_list_present_remove_list_any(self):
        self.assertRewrittenEqual(Union[List[str], List[Any]], List[str])
        self.assertRewrittenEqual(
            Union[str, List[str], List[Any]], Union[str, List[str]]
        )

    def test_rewrite_dicts(self):
        self.assertRewrittenEqual(Union[Dict[Any, Any], Dict[str, Any]], Dict[str, Any])
        self.assertRewrittenEqual(
            Union[Dict[Any, Any], Dict[str, Any], str], Union[Dict[str, Any], str]
        )
        self.assertRewrittenEqual(
            Union[Tuple[str, Dict[str, str]], Tuple[str, Dict[Any, Any]]],
            Tuple[str, Dict[str, str]],
        )


class _A(object):
    pass


class A(_A):
    pass


class B(A):
    pass


class C(B):
    pass


class MyString(str):
    pass


class TestSimplifyUnionOfListsSets(BaseRewriterTestCase):
    def setUp(self):
        self.rewriter = SimplifyByMROTraversing()

    def test_list_of_subclass_converted_into_list_of_baseclass_if_present(self):
        self.assertRewrittenEqual(Union[List[A], List[B]], List[A])
        self.assertRewrittenEqual(Union[List[A], List[B], str], Union[List[A], str])
        self.assertRewrittenEqual(Union[Set[A], Set[B]], Set[A])
        self.assertRewrittenEqual(Union[Dict[str, A], Dict[str, B]], Dict[str, A])
        self.assertRewrittenEqual(Union[Dict[A, str], Dict[B, str]], Dict[A, str])
        self.assertRewrittenEqual(Union[Dict[A, A], Dict[B, B]], Dict[A, A])
        self.assertRewrittenEqual(
            Union[Dict[A, A], Dict[B, B], Dict[C, B], Dict[C, C]], Dict[A, A]
        )

    def test_do_not_rewrite_if_base_in_stdlib(self):
        self.assertRewrittenEqual(
            Union[List[str], List[MyString]], Union[List[str], List[MyString]]
        )

    def test_rewrite_more_than_two_elements(self):
        self.assertRewrittenEqual(
            Union[List[A], List[B], List[str]], Union[List[A], List[str]]
        )
        self.assertRewrittenEqual(Union[List[A], List[B], List[C]], List[A])

    def test_choose_deepest_common_baseclass_if_not_direct_subclasses(self):
        class B2(A):
            pass

        class C2(B2):
            pass

        self.assertRewrittenEqual(Union[C, C2], A)
        self.assertRewrittenEqual(Union[List[C], List[C2]], List[A])

    def test_union_of_tuples_if_subclass_into_tuple(self):
        self.assertRewrittenEqual(Union[Tuple[A], Tuple[B]], Tuple[A])
        self.assertRewrittenEqual(
            Union[Tuple[str, int, A], Tuple[str, int, B], Tuple[str, int, C]],
            Tuple[str, int, A],
        )


class TestSimplifyIntFloat(BaseRewriterTestCase):
    def setUp(self):
        self.rewriter = SimplifyIntFloat()

    def test_no_float_leave_as_is(self):
        self.assertRewrittenEqual(Union[int, str], Union[int, str])
        self.assertRewrittenEqual(Union[float, str], Union[float, str])

    def test_if_int_and_float_present_simplify_to_float(self):
        self.assertRewrittenEqual(Union[int, float], float)
        self.assertRewrittenEqual(Union[float, int], float)
        self.assertRewrittenEqual(Union[int, float, List[str]], Union[float, List[str]])


class TestUnionOfTypes(BaseRewriterTestCase):
    def setUp(self):
        self.rewriter = SimplifyUnionOfTypes()

    def test_union_of_types_is_type_of_union(self):
        self.assertRewrittenEqual(Union[Type[str], Type[int]], Type[Union[str, int]])
        self.assertRewrittenEqual(
            Union[Type[str], Type[int], str, List[str]],
            Union[Type[Union[str, int]], str, List[str]],
        )


class TestDepthFirstTraverser(BaseRewriterTestCase):
    def setUp(self):
        self.rewriter = DepthFirstTypeTraverser(SimplifyIntFloat())

    def test_if_just_element_returns_it(self):
        self.assertRewrittenEqual(int, int)
        self.assertRewrittenEqual(str, str)
        self.assertRewrittenEqual(List, List)
        self.assertRewrittenEqual(Dict, Dict)
        self.assertRewrittenEqual(Tuple[int, int, str], Tuple[int, int, str])

    def test_nested_unions_width_generics(self):
        self.assertRewrittenEqual(List[Union[int, float]], List[float])
        self.assertRewrittenEqual(
            List[
                Union[Tuple[float, float], Tuple[Union[float, int], Union[float, int]]]
            ],
            List[Tuple[float, float]],
        )


class TestEverything(BaseRewriterTestCase):
    def setUp(self):
        two_element_transformers = [simplify_types,
                                    simplify_int_float,
                                    FindAcceptableCommonBase(allowed_bases_prefixes=['django']),
                                    remove_empty_container]
        self.rewriter = DepthFirstTypeTraverser(
            union_rewriter=TwoElementUnionRewriter(
                two_element_transformers=[
                    *two_element_transformers,
                    SimplifyGenerics(two_element_transformers=two_element_transformers)
                ]
            ))

    def test_small_stuff(self):
        self.assertRewrittenEqual(Union[
                                      Tuple[str, Dict[str, Union[bytes, str]]],
                                      Tuple[str, Dict[Any, Any]]
                                  ],
                                  Tuple[str, Dict[str, Union[bytes, str]]])

    def test_big_stuff(self):
        original = Iterator[
            Union[
                Tuple[str, Dict[Any, Any], LazyStream],  #
                Tuple[
                    str,
                    Dict[
                        str,
                        Union[Tuple[str, Dict[str, bytes]], Tuple[str, Dict[Any, Any]]],
                    ],
                    LazyStream,
                ],
                Tuple[str, Dict[str, Tuple[str, Dict[str, bytes]]], LazyStream],
                Tuple[
                    str,
                    Dict[
                        str,
                        Union[
                            Tuple[str, Dict[str, Union[bytes, str]]],
                            Tuple[str, Dict[Any, Any]],
                        ],
                    ],
                    LazyStream,
                ],
                Tuple[
                    str,
                    Dict[
                        str,
                        Union[Tuple[str, Dict[str, str]], Tuple[str, Dict[Any, Any]]],
                    ],
                    LazyStream,
                ],
            ]
        ]
        final = Iterator[
            Tuple[str,
                  Dict[str,
                       Tuple[str,
                             Dict[str, Union[str, bytes]]]],
                  LazyStream]]

        self.assertRewrittenEqual(original, final)

    def test_common_bases(self):
        pass