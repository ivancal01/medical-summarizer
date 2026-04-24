# utils/parser_manager.py
"""
Parser manager for handling multiple document parsers.
Provides automatic parser selection and unified interface.
"""

from typing import List, Optional, Dict, Any
from parsers.base_parser import BaseParser, ParserResult, ParserError
from parsers.pdf_parser import PDFParser
from parsers.pubmed_parser import PubMedParser
from parsers.arxiv_parser import ArXivParser


class ParserManager:
    """
    Manager class for handling multiple document parsers.
    Automatically selects the appropriate parser based on input source.
    """
    
    def __init__(self, pubmed_email: str = "example@example.com", 
                 pubmed_api_key: Optional[str] = None):
        """
        Initialize parser manager.
        
        Args:
            pubmed_email: Email for NCBI API (required by policy).
            pubmed_api_key: Optional NCBI API key for higher rate limits.
        """
        self.parsers: List[BaseParser] = [
            PubMedParser(api_key=pubmed_api_key, email=pubmed_email),
            ArXivParser(),
            PDFParser()
        ]
        
        self._parser_cache: Dict[str, BaseParser] = {}
    
    def get_parser(self, source: str) -> Optional[BaseParser]:
        """
        Find appropriate parser for the given source.
        
        Args:
            source: Source identifier (URL, file path, DOI, etc.)
            
        Returns:
            Optional[BaseParser]: Appropriate parser or None.
        """
        # Check cache first
        if source in self._parser_cache:
            return self._parser_cache[source]
        
        # Try each parser
        for parser in self.parsers:
            if parser.can_parse(source):
                self._parser_cache[source] = parser
                return parser
        
        return None
    
    def parse(self, source: str) -> ParserResult:
        """
        Parse document using appropriate parser.
        
        Args:
            source: Source identifier.
            
        Returns:
            ParserResult: Parsed document content and metadata.
            
        Raises:
            ParserError: If no suitable parser found or parsing fails.
        """
        parser = self.get_parser(source)
        
        if not parser:
            raise ParserError(
                f"No suitable parser found for source: {source}. "
                f"Supported formats: PDF files, PubMed (PMID/DOI/URL), arXiv (ID/URL)",
                source
            )
        
        return parser.parse(source)
    
    def parse_with_fallback(self, source: str) -> ParserResult:
        """
        Parse document with fallback to other parsers.
        
        Args:
            source: Source identifier.
            
        Returns:
            ParserResult: Parsed document content and metadata.
            
        Raises:
            ParserError: If all parsers fail.
        """
        errors = []
        
        # Try primary parser
        parser = self.get_parser(source)
        if parser:
            try:
                return parser.parse(source)
            except ParserError as e:
                errors.append(f"{parser.__class__.__name__}: {e.message}")
        
        # Try all other parsers as fallback
        for parser in self.parsers:
            try:
                return parser.parse(source)
            except ParserError as e:
                errors.append(f"{parser.__class__.__name__}: {e.message}")
            except Exception as e:
                errors.append(f"{parser.__class__.__name__}: {str(e)}")
        
        raise ParserError(
            f"All parsers failed. Errors: {'; '.join(errors)}",
            source
        )
    
    def get_supported_formats(self) -> Dict[str, str]:
        """
        Get list of supported formats.
        
        Returns:
            Dict[str, str]: Format descriptions.
        """
        return {
            'pdf': 'PDF documents (.pdf files)',
            'pubmed': 'PubMed articles (PMID, DOI, or pubmed.ncbi.nlm.nih.gov URLs)',
            'arxiv': 'arXiv preprints (arXiv ID or arxiv.org URLs)'
        }
    
    def add_parser(self, parser: BaseParser) -> None:
        """
        Add custom parser to the manager.
        
        Args:
            parser: Custom parser instance.
        """
        self.parsers.append(parser)
        self._parser_cache.clear()  # Clear cache
    
    def remove_parser(self, parser_class: type) -> bool:
        """
        Remove parser by class type.
        
        Args:
            parser_class: Parser class to remove.
            
        Returns:
            bool: True if parser was removed.
        """
        for i, parser in enumerate(self.parsers):
            if isinstance(parser, parser_class):
                del self.parsers[i]
                self._parser_cache.clear()
                return True
        return False
