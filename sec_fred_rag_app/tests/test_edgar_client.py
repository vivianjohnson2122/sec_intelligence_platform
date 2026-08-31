"""
Unit tests for ingestion/edgar_client.py wrapper detection and filename parsing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.edgar_client import EdgarClient

AMZN_WRAPPER_SNIPPET = """
<SEC-DOCUMENT>0001018724-24-000008-index.html : 20240202
<SEC-HEADER>0001018724-24-000008.hdr.sgml : 20240202
ACCESSION NUMBER: 0001018724-24-000008
CONFORMED SUBMISSION TYPE: 10-K
<DOCUMENT>
<TYPE>10-K
<SEQUENCE>1
<FILENAME>amzn-20231231.htm
<DESCRIPTION>10-K
<TEXT>
 Document 1 - file: amzn-20231231.htm
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-10.7
<SEQUENCE>2
<FILENAME>amzn-20231231xex107.htm
<DESCRIPTION>EX-10.7
<TEXT>
 Document 2 - file: amzn-20231231xex107.htm
</DOCUMENT>
</SEC-DOCUMENT>
"""


class TestEdgarClientHelpers:
    def setup_method(self) -> None:
        self.client = EdgarClient()

    def test_detects_submission_wrapper(self) -> None:
        assert self.client._is_submission_wrapper(AMZN_WRAPPER_SNIPPET)

    def test_does_not_flag_real_html(self) -> None:
        html = "<html><body><h1>Item 1. Business</h1><p>Revenue grew.</p></body></html>"
        assert not self.client._is_submission_wrapper(html)

    def test_extracts_primary_filename_from_wrapper(self) -> None:
        filename = self.client._extract_primary_filename(AMZN_WRAPPER_SNIPPET, "10-K")
        assert filename == "amzn-20231231.htm"

    def test_skips_exhibits_and_xbrl_pages(self) -> None:
        assert self.client._is_non_primary_document("amzn-20231231xex107.htm")
        assert self.client._is_non_primary_document("R12.htm")
        assert self.client._is_non_primary_document("0001018724-24-000008-index.htm")
        assert not self.client._is_non_primary_document("amzn-20231231.htm")
