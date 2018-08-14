from typing import List, Tuple, Union, Any, Optional

from typeline.rewriters.tests.test_generics import BaseRewriterTestCase
from typeline.rewriters.transformers import TwoElementUnionRewriter, remove_empty_container


class TestDepthFirstRewriter(BaseRewriterTestCase):
    def setUp(self):
        self.rewriter = TwoElementUnionRewriter(
            two_element_transformers=[
                remove_empty_container
            ])

    def test_simplified(self):
        self.assertRewrittenEqual(List[Tuple[str,
                                             Union[
                                                 List[Any],
                                                 List[Optional[Union[float, str]]],
                                                 List[Optional[Union[int, str]]],
                                                 List[memoryview],
                                                 Tuple,
                                             ]]],
                                  List[Tuple[str, Union[List[Optional[Union[float, str]]],
                                                        Tuple]]])
