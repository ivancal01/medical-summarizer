# parsers/pubmed_parser.py
"""
PubMed article parser using NCBI E-utilities API.
"""

import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from urllib.request import urlopen, Request
from urllib.parse import quote
import json

from .base_parser import BaseParser, ParserResult, ParserError


class PubMedParser(BaseParser):
    """
    Parser for PubMed articles using NCBI E-utilities API.
    Supports searching by PMID, DOI, or keywords.
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, api_key: Optional[str] = None, email: str = "example@example.com"):
        """
        Initialize PubMed parser.
        
        Args:
            api_key: NCBI API key (optional, increases rate limits).
            email: Email address for NCBI (required by their policy).
        """
        super().__init__()
        self.api_key = api_key
        self.email = email
        self._rate_limit_delay = 0.34  # 3 requests per second without API key
    
    def can_parse(self, source: str) -> bool:
        """
        Check if source is a PubMed identifier.
        
        Args:
            source: PMID, DOI, or PubMed URL.
            
        Returns:
            bool: True if source appears to be PubMed-related.
        """
        # PMID (numeric)
        if source.isdigit() and len(source) <= 9:
            return True
        
        # PubMed URL
        if 'pubmed.ncbi.nlm.nih.gov' in source.lower():
            return True
        
        # DOI starting with 10.
        if source.startswith('10.') and '/' in source:
            return True
        
        # PMID: prefix
        if source.lower().startswith('pmid:'):
            return True
        
        return False
    
    def parse(self, source: str) -> ParserResult:
        """
        Parse PubMed article.
        
        Args:
            source: PMID, DOI, or PubMed URL.
            
        Returns:
            ParserResult: Article content and metadata.
            
        Raises:
            ParserError: If parsing fails.
        """
        try:
            pmid = self._extract_pmid(source)
            
            if not pmid:
                raise ParserError("Could not extract PMID from source", source)
            
            # Fetch article data
            article_data = self._fetch_article(pmid)
            
            if not article_data:
                raise ParserError(f"No article found for PMID {pmid}", source)
            
            # Extract components
            title = self._extract_field(article_data, 'ArticleTitle')
            abstract = self._extract_abstract(article_data)
            authors = self._extract_authors(article_data)
            journal = self._extract_journal(article_data)
            pub_date = self._extract_pub_date(article_data)
            
            # Combine full text (abstract + available text)
            full_text = self._extract_full_text(article_data)
            content = f"{title}\n\n{abstract}\n\n{full_text}" if full_text else f"{title}\n\n{abstract}"
            
            result = ParserResult(
                content=content.strip(),
                title=title,
                authors=authors,
                abstract=abstract,
                metadata={
                    'pmid': pmid,
                    'journal': journal,
                    'pub_date': pub_date,
                    'doi': self._extract_field(article_data, 'ELocationID', {'EIdType': 'doi'}),
                    'source': source,
                    'parser_type': 'pubmed'
                }
            )
            
            if not self.validate_content(result.content):
                raise ParserError("Extracted content is too short or invalid", source)
            
            return result
            
        except Exception as e:
            if isinstance(e, ParserError):
                raise
            raise ParserError(f"Failed to parse PubMed article: {str(e)}", source)
    
    def _extract_pmid(self, source: str) -> Optional[str]:
        """
        Extract PMID from various source formats.
        
        Args:
            source: Input source string.
            
        Returns:
            Optional[str]: Extracted PMID or None.
        """
        source = source.strip()
        
        # Direct PMID
        if source.isdigit() and len(source) <= 9:
            return source
        
        # PMID: prefix
        match = re.search(r'PMID[:\s]*(\d+)', source, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # PubMed URL
        match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', source)
        if match:
            return match.group(1)
        
        # DOI - need to resolve via API
        if source.startswith('10.') and '/' in source:
            return self._resolve_doi(source)
        
        return None
    
    def _resolve_doi(self, doi: str) -> Optional[str]:
        """
        Resolve DOI to PMID.
        
        Args:
            doi: DOI string.
            
        Returns:
            Optional[str]: PMID or None.
        """
        try:
            url = f"{self.BASE_URL}/esearch.fcgi?db=pubmed&term={quote(doi)}[DOI]&retmode=json"
            if self.api_key:
                url += f"&api_key={self.api_key}"
            
            response = urlopen(url, timeout=10)
            data = json.loads(response.read().decode())
            
            ids = data.get('esearchresult', {}).get('idlist', [])
            return ids[0] if ids else None
            
        except Exception:
            return None
    
    def _build_url(self, endpoint: str, params: Dict[str, str]) -> str:
        """
        Build NCBI E-utilities URL.
        
        Args:
            endpoint: API endpoint.
            params: Query parameters.
            
        Returns:
            str: Complete URL.
        """
        params['email'] = self.email
        if self.api_key:
            params['api_key'] = self.api_key
        
        query = '&'.join(f"{k}={v}" for k, v in params.items())
        return f"{self.BASE_URL}/{endpoint}?{query}"
    
    def _fetch_article(self, pmid: str) -> Optional[ET.Element]:
        """
        Fetch article XML from PubMed.
        
        Args:
            pmid: PubMed ID.
            
        Returns:
            Optional[ET.Element]: Parsed XML element or None.
        """
        try:
            url = self._build_url(
                "efetch.fcgi",
                {'db': 'pubmed', 'id': pmid, 'rettype': 'xml', 'retmode': 'xml'}
            )
            
            response = urlopen(url, timeout=15)
            xml_data = response.read()
            
            root = ET.fromstring(xml_data)
            pubmed_article = root.find('.//PubmedArticle')
            
            return pubmed_article if pubmed_article is not None else None
            
        except Exception as e:
            print(f"Error fetching article: {e}")
            return None
    
    def _extract_field(self, article: ET.Element, field_name: str, 
                      attrib_filter: Optional[Dict[str, str]] = None) -> str:
        """
        Extract field from XML.
        
        Args:
            article: XML element.
            field_name: Field name to find.
            attrib_filter: Optional attribute filter.
            
        Returns:
            str: Field value or empty string.
        """
        for elem in article.iter(field_name):
            if attrib_filter:
                if all(elem.get(k) == v for k, v in attrib_filter.items()):
                    return (elem.text or '').strip()
            else:
                return (elem.text or '').strip()
        
        return ""
    
    def _extract_abstract(self, article: ET.Element) -> str:
        """
        Extract abstract from XML.
        
        Args:
            article: XML element.
            
        Returns:
            str: Abstract text.
        """
        abstract_parts = []
        
        for abstract in article.iter('Abstract'):
            for child in abstract:
                if child.tag == 'AbstractText':
                    label = child.get('Label', '')
                    text = (child.text or '').strip()
                    if label and text:
                        abstract_parts.append(f"**{label}**: {text}")
                    elif text:
                        abstract_parts.append(text)
        
        return '\n\n'.join(abstract_parts)
    
    def _extract_authors(self, article: ET.Element) -> List[str]:
        """
        Extract authors from XML.
        
        Args:
            article: XML element.
            
        Returns:
            List[str]: List of author names.
        """
        authors = []
        
        for author in article.iter('Author'):
            last_name = author.findtext('LastName', '')
            fore_name = author.findtext('ForeName', '')
            
            if last_name:
                if fore_name:
                    authors.append(f"{fore_name} {last_name}")
                else:
                    authors.append(last_name)
        
        # Fallback to collective name
        if not authors:
            for collective in article.iter('CollectiveName'):
                if collective.text:
                    authors.append(collective.text.strip())
        
        return authors
    
    def _extract_journal(self, article: ET.Element) -> str:
        """
        Extract journal name.
        
        Args:
            article: XML element.
            
        Returns:
            str: Journal name.
        """
        for journal in article.iter('Journal'):
            title = journal.findtext('Title', '')
            if title:
                return title
        return ""
    
    def _extract_pub_date(self, article: ET.Element) -> str:
        """
        Extract publication date.
        
        Args:
            article: XML element.
            
        Returns:
            str: Publication date.
        """
        for pub_date in article.iter('PubDate'):
            year = pub_date.findtext('Year', '')
            month = pub_date.findtext('Month', '')
            day = pub_date.findtext('Day', '')
            medline_date = pub_date.findtext('MedlineDate', '')
            
            if year:
                parts = [year]
                if month:
                    parts.append(month)
                if day:
                    parts.append(day)
                return ' '.join(parts)
            elif medline_date:
                return medline_date
        
        return ""
    
    def _extract_full_text(self, article: ET.Element) -> str:
        """
        Extract available full text sections.
        
        Args:
            article: XML element.
            
        Returns:
            str: Full text content.
        """
        text_parts = []
        
        # Look for various text sections
        for section in article.iter('Section'):
            title = section.findtext('SectionTitle', '')
            paragraphs = section.findall('.//Paragraph')
            
            if paragraphs:
                if title:
                    text_parts.append(f"\n## {title}\n")
                for p in paragraphs:
                    if p.text:
                        text_parts.append(p.text.strip())
        
        return '\n\n'.join(text_parts)
