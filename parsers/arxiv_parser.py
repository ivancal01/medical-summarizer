# parsers/arxiv_parser.py
"""
arXiv preprint parser using arXiv API.
"""

import re
import json
from typing import List, Dict, Any, Optional
from urllib.request import urlopen, Request
from urllib.parse import quote
import xml.etree.ElementTree as ET

from .base_parser import BaseParser, ParserResult, ParserError


class ArXivParser(BaseParser):
    """
    Parser for arXiv preprints using the arXiv API.
    Supports searching by arXiv ID, DOI, or URL.
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        """Initialize arXiv parser."""
        super().__init__()
    
    def can_parse(self, source: str) -> bool:
        """
        Check if source is an arXiv identifier.
        
        Args:
            source: arXiv ID, DOI, or arXiv URL.
            
        Returns:
            bool: True if source appears to be arXiv-related.
        """
        # arXiv URL
        if 'arxiv.org' in source.lower():
            return True
        
        # arXiv ID patterns (e.g., 2103.12345, hep-th/9901001, 2301.01234)
        if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', source):
            return True
        
        # Old-style arXiv IDs (e.g., hep-th/9901001)
        if re.match(r'^[a-z-]+/\d{7}$', source):
            return True
        
        # arXiv: prefix
        if source.lower().startswith('arxiv:'):
            return True
        
        return False
    
    def parse(self, source: str) -> ParserResult:
        """
        Parse arXiv preprint.
        
        Args:
            source: arXiv ID, DOI, or URL.
            
        Returns:
            ParserResult: Article content and metadata.
            
        Raises:
            ParserError: If parsing fails.
        """
        try:
            arxiv_id = self._extract_arxiv_id(source)
            
            if not arxiv_id:
                raise ParserError("Could not extract arXiv ID from source", source)
            
            # Fetch paper data
            paper_data = self._fetch_paper(arxiv_id)
            
            if not paper_data:
                raise ParserError(f"No paper found for arXiv ID {arxiv_id}", source)
            
            # Extract components
            title = self._get_text(paper_data, '{http://www.w3.org/2005/Atom}title')
            summary = self._get_text(paper_data, '{http://www.w3.org/2005/Atom}summary')
            authors = self._extract_authors(paper_data)
            published = self._get_text(paper_data, '{http://www.w3.org/2005/Atom}published')
            categories = self._extract_categories(paper_data)
            doi = self._get_doi(paper_data)
            
            # Clean summary (abstract)
            abstract = self._clean_text(summary)
            
            # For arXiv, we primarily have the abstract
            # Full text would require PDF download
            content = f"{title}\n\n{abstract}"
            
            result = ParserResult(
                content=content.strip(),
                title=title.strip(),
                authors=authors,
                abstract=abstract,
                metadata={
                    'arxiv_id': arxiv_id,
                    'published': published,
                    'categories': categories,
                    'doi': doi,
                    'source': source,
                    'parser_type': 'arxiv'
                }
            )
            
            if not self.validate_content(result.content):
                raise ParserError("Extracted content is too short or invalid", source)
            
            return result
            
        except Exception as e:
            if isinstance(e, ParserError):
                raise
            raise ParserError(f"Failed to parse arXiv paper: {str(e)}", source)
    
    def _extract_arxiv_id(self, source: str) -> Optional[str]:
        """
        Extract arXiv ID from various source formats.
        
        Args:
            source: Input source string.
            
        Returns:
            Optional[str]: Extracted arXiv ID or None.
        """
        source = source.strip().lower()
        
        # Remove arXiv: prefix
        if source.startswith('arxiv:'):
            source = source[6:]
        
        # Extract from URL
        url_patterns = [
            r'arxiv\.org/abs/([^\s\?#/]+)',
            r'arxiv\.org/pdf/([^\s\?#/]+)',
            r'arxiv\.org/ftp/[^/]+/papers/([^/]+)/.*',
        ]
        
        for pattern in url_patterns:
            match = re.search(pattern, source)
            if match:
                arxiv_id = match.group(1)
                # Remove version suffix for lookup
                arxiv_id = arxiv_id.split('v')[0]
                return self._normalize_arxiv_id(arxiv_id)
        
        # Direct ID
        source = source.split('v')[0]  # Remove version
        
        # New style: YYMM.NNNNN
        if re.match(r'^\d{4}\.\d{4,5}$', source):
            return source
        
        # Old style: category/NNNNNNN
        if re.match(r'^[a-z-]+/\d{7}$', source):
            return source
        
        return None
    
    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """
        Normalize arXiv ID to standard format.
        
        Args:
            arxiv_id: Raw arXiv ID.
            
        Returns:
            str: Normalized arXiv ID.
        """
        arxiv_id = arxiv_id.strip()
        
        # Already in new format
        if re.match(r'^\d{4}\.\d{4,5}$', arxiv_id):
            return arxiv_id
        
        # Convert old format to new if possible
        # This is a simplification - full conversion requires a lookup table
        return arxiv_id
    
    def _build_query_url(self, arxiv_id: str) -> str:
        """
        Build arXiv API query URL.
        
        Args:
            arxiv_id: arXiv ID.
            
        Returns:
            str: Query URL.
        """
        # Search by ID
        id_search = f'id:{arxiv_id}'
        
        params = {
            'search_query': id_search,
            'start': 0,
            'max_results': 1,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        query = '&'.join(f"{k}={quote(v)}" for k, v in params.items())
        return f"{self.BASE_URL}?{query}"
    
    def _fetch_paper(self, arxiv_id: str) -> Optional[ET.Element]:
        """
        Fetch paper data from arXiv API.
        
        Args:
            arxiv_id: arXiv ID.
            
        Returns:
            Optional[ET.Element]: XML entry element or None.
        """
        try:
            url = self._build_query_url(arxiv_id)
            
            response = urlopen(url, timeout=15)
            xml_data = response.read()
            
            root = ET.fromstring(xml_data)
            
            # Find the first entry
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('atom:entry', ns)
            
            return entry
            
        except Exception as e:
            print(f"Error fetching paper: {e}")
            return None
    
    def _get_text(self, element: ET.Element, tag: str) -> str:
        """
        Get text content from XML element.
        
        Args:
            element: XML element.
            tag: Tag name to find.
            
        Returns:
            str: Text content or empty string.
        """
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return ""
    
    def _extract_authors(self, entry: ET.Element) -> List[str]:
        """
        Extract authors from XML entry.
        
        Args:
            entry: XML entry element.
            
        Returns:
            List[str]: List of author names.
        """
        authors = []
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for author in entry.findall('atom:author', ns):
            name_elem = author.find('atom:name', ns)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
        
        return authors
    
    def _extract_categories(self, entry: ET.Element) -> List[str]:
        """
        Extract categories/tags from XML entry.
        
        Args:
            entry: XML entry element.
            
        Returns:
            List[str]: List of categories.
        """
        categories = []
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for category in entry.findall('atom:category', ns):
            term = category.get('term')
            if term:
                categories.append(term)
        
        return categories
    
    def _get_doi(self, entry: ET.Element) -> Optional[str]:
        """
        Extract DOI from XML entry.
        
        Args:
            entry: XML entry element.
            
        Returns:
            Optional[str]: DOI or None.
        """
        # DOI is often in the doi tag or in links
        doi_elem = entry.find('{http://arxiv.org/schemas/atom}doi')
        if doi_elem is not None and doi_elem.text:
            return doi_elem.text.strip()
        
        # Check links for DOI
        for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
            if link.get('title') == 'doi':
                href = link.get('href')
                if href and 'doi.org' in href:
                    return href.split('doi.org/')[-1]
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw text.
            
        Returns:
            str: Cleaned text.
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common formatting issues
        text = re.sub(r' ([,.:;])', r'\1', text)
        
        return text.strip()
