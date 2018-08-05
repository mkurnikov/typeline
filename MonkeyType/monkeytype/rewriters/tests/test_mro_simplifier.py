from typing import List, Union, Any, Dict, Tuple

from monkeytype.rewriters.tests.module1 import A1, B1
from monkeytype.rewriters.tests.module2 import A2, D2
from monkeytype.rewriters.tests.test_generics import BaseRewriterTestCase
from monkeytype.rewriters.transformers import MroSimplifier


class TestMroSimplifier(BaseRewriterTestCase):
    def setUp(self):
        self.rewriter = MroSimplifier(filtered_modules=['monkeytype.rewriters.tests.module2'])

    def test_smoke1(self):
        self.assertRewrittenEqual(List[Any], List[Any])
        self.assertRewrittenEqual(A2, Any)

    def test_smoke2(self):
        self.assertRewrittenEqual(List[Union[A1, A2]], List[A1])
        self.assertRewrittenEqual(Union[List[A1], List[A2]], Union[List[A1], List[Any]])

    def test_smoke3(self):
        self.assertRewrittenEqual(List[Union[A1, D2]], List[Union[A1, B1]])
        self.assertRewrittenEqual(Union[List[Tuple[A2, str, str]], str],
                                  Union[List[Tuple[Any, str, str]], str])
        self.assertRewrittenEqual(List[Union[List[Tuple[A2, str, str]], str]],
                                  List[Union[List[Tuple[Any, str, str]], str]])
        self.assertRewrittenEqual(Union[List[Union[List[Tuple[A2, str, str]], str]],
                                        List[Union[List[Tuple[D2, int, str]], str]]],
                                  Union[List[Union[List[Tuple[Any, str, str]], str]],
                                        List[Union[List[Tuple[B1, int, str]], str]]])

    def test_smoke4(self):
        self.assertRewrittenEqual(List[A2], List[Any])
        self.assertRewrittenEqual(Dict[str, A2], Dict[str, Any])
