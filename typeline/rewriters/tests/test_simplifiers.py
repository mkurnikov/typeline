from typing import Union, Dict, List, MutableMapping

from django.core.exceptions import ValidationError
from django.forms.utils import ErrorDict

from typeline.rewriters.tests.test_generics import BaseRewriterTestCase
from typeline.rewriters.transformers import TwoElementUnionRewriter, simplify_to_abstract_class


class TestABCSimplifier(BaseRewriterTestCase):
    def setUp(self):
        self.rewriter = TwoElementUnionRewriter(two_element_transformers=[
            simplify_to_abstract_class
        ])

    def test_simplify_userdict_to_mutable_mapping(self):
        self.assertRewrittenEqual(Union[Dict[str, List[ValidationError]], List[str]],
                                  Union[Dict[str, List[ValidationError]], List[str]])

        self.assertRewrittenEqual(Union[Dict[str, List[ValidationError]], ErrorDict],
                                  Dict[str, List[ValidationError]])