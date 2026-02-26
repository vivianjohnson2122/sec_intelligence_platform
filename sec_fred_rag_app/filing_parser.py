"""
Parse raw SEC filing HTML into structured sections 

10-K filings follow a standard structure: 

    Item 1 - Business
    Item 1A - Risk Factors 
    Item 7 - Management's Discussion and Analysis (MD&A)
    Item 7A - Quantitative Disclousers About Market Risk
    Item 8 - Financial Statements 

We extract these sections by searching for their standard headers,
which gives us semantically meaningful chunks for RAG retrieval 
"""

import re # regex 
from bs4 import BeautifulSoup
from typing import Optional
from spacy.lang.en import English


# section patterns to find in filings 
# each tuple: (section_key, display_name, list_of_header_patterns)
SECTION_PATTERNS_10K = [
    (
        "business",
        "Business Overview",
        [r"item\s*1\b(?!a)", r"description of business"]
    ),
    (
        "risk_factors",
        "Risk Factors",
        [r"item\s*1a", r"risk factors"],
    ),
    (
        "mda",
        "Management's Discussion & Analysis",
        [r"item\s*7\b(?!a)", r"management.{0,10}discussion", r"md&a"],
    ),
    (
        "market_risk",
        "Quantitative Market Risk Disclosures",
        [r"item\s*7a", r"quantitative.*market risk"],
    ),
    (
        "financials",
        "Financial Statements",
        [r"item\s*8\b", r"financial statements"],
    ),
    (
        "mda",
        "Management's Discussion & Analysis",
        [r"item\s*2\b", r"management.{0,10}discussion"],
    ),
    (
        "market_risk",
        "Quantitative Market Risk Disclosures",
        [r"item\s*3\b", r"quantitative.*market risk"],
    ),
    (
        "financials",
        "Financial Statements",
        [r"item\s*1\b", r"financial statements"],
    ),
]


class FilingParser:
    """
    Parse a raw SEC filing (HTML String) into named sections

    Usage: 
        parser = FilingParser()
        sections = parser.parse(html_text, form_type = "10-K)
        # sections = {"mda": "...", "risk_factors: "...", ...}
    """


    def html_to_text(self,
                     html: str) -> str:
        """
        Strip HTML tags and normalize whitespace 
        """

        soup = BeautifulSoup(html,
                             "xml") # for parsing html 
        # remove script / style tags 
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        
        text = soup.get_text(separator=" ")
        # collapse whitespace
        text = re.sub(r"[ \t]+", " ", text) # remove multiple space/tabs
        text = re.sub(r"\n{3,}", "\n\n", text) # remove excessive spacing 
        return text.strip()


    def extract_sections(self,
                        text: str,
                        patterns: list[tuple]) -> dict[str, str]:
        """
        Find section boundaries by searching for header patterns,
        then slice the text between consecutive headers 
        """

        text_lower = text.lower()
        found_positions = [] # (position, section_key, display_name) where section starts 

        for section_key, display_name, header_patterns in patterns:
            for pattern in header_patterns:
                matches = list(re.finditer(pattern, text_lower)) # try all header patterns
                if matches:
                    # take last match store position
                    pos = matches[-1].start() 
                    found_positions.append((pos, section_key, display_name))
                    break
        
        if not found_positions:
            return {}
        
        # sort by position in document 
        found_positions.sort(key = lambda x: x[0])

        sections = {}
        for i, (pos, key, name) in enumerate(found_positions):
            # section ends at the start of the next section (or 30k chars limit)
            if i + 1 < len(found_positions):
                end = found_positions[i + 1][0]
            else:
                end = pos + 30_000

            section_text = text[pos:end].strip()

            # skip very short extractions (likely false positives)
            if len(section_text) > 200:
                sections[key] = section_text
                print(f"[Parser] Extracted '{name}': {len(section_text):,} chars")

        return sections            


    def parse(self,
              raw_html: str,
              form_type: str="10-K") -> dict[str, str]:
        """
        Parse HTML into a dictionary of {section_key: cleaned_text}
        Falls back to full text if section parsing fails 
        """
        
        # 1. clean raw html to text 
        clean_text = self.html_to_text(raw_html)

        # extract sections using form type patterns 
        sections = self.extract_sections(clean_text,
                                        patterns=SECTION_PATTERNS_10K)
        
        # return whole doc if can't extract sections 
        if not sections:
            print("[Parser] Section extraction failed, using full text fallback")
            sections = {"full_text": clean_text[:50000]} # cap at 50k chars 
        
        return sections 


def chunk_text(text: str,
               chunk_size_chars: int=800,
               overlap_sentences: int=2) -> list[str]:
    """
    Split text into overlapping chunks for embedding. 

    Default size of chunk_size_char is 800 as small embedding models perform better
    on chunks between 100-256 tokens more for precise answers. 
    Sentence level overlap

    Args: 
        text: Input text
        chunk_size: Target chars per chunk 
        chunk_overlap: Overlap between chunks to preserve context 

    Returns: 
        List of text chunks 
    """
    # storing results 
    chunks = []
    curr_chunk = []
    curr_len = 0

    nlp = English()
    # sentencizer pipeline to turn text into sentences 
    nlp.add_pipe("sentencizer")
    sentences = list(nlp(text).sents)
    # ensure all strings 
    sentences = [str(s) for s in sentences]

    for sentence in sentences: 
        s_len = len(sentence)
        # check if can add sentence to curr_chunk 
        if curr_len + s_len > chunk_size_chars and curr_chunk:
            chunks.append(" ".join(curr_chunk))
            
            # sentence level overlap to preserve sentences
            curr_chunk = curr_chunk[-overlap_sentences]
            curr_len = sum(len(s) for s in curr_chunk)
        else:
            curr_chunk.append(sentence)
            curr_len += s_len
    
    if curr_chunk:
        chunks.append(" ".join(curr_chunk))
    
    return [c for c in chunks if len(c) > 50]

