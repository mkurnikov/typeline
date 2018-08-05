from typing import Union, Tuple, List, Any, Optional

from django_stubs_root.generator.django_git.django.db.models.expressions import Expression, Col
from monkeytype.rewriters import transformers
from monkeytype.rewriters.tests.test_generics import BaseRewriterTestCase
from monkeytype.rewriters.transformers import SimplifyTuples, TwoElementUnionRewriter


class TestTuples(BaseRewriterTestCase):
    def setUp(self):
        # self.rewriter = SimplifyTuples(two_element_transformers=[transformers.find_acceptable_common_base])
        self.rewriter = TwoElementUnionRewriter(
            two_element_transformers=[
                SimplifyTuples(two_element_transformers=[transformers.find_acceptable_common_base])
            ])

    def test_smoke(self):
        self.assertRewrittenEqual(Union[Tuple[Expression, Tuple[str, List[Any]], str],
                                        Tuple[Col, Tuple[str, List[Any]], None]],
                                  Tuple[Expression, Tuple[str, List[Any]], Optional[str]])

    def test_not_simplified(self):
        self.assertRewrittenEqual(Union[Tuple[Expression, Tuple[str, List[Any], str], str],
                                        Tuple[Col, Tuple[int, List[Any], int], None]],
                                  Union[Tuple[Expression, Tuple[str, List[Any], str], str],
                                        Tuple[Col, Tuple[int, List[Any], int], None]])
