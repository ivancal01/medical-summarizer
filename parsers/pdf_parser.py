# parsers/pdf_parser.py
"""
PDF document parser using PyMuPDF (fitz).
"""

import os
import re
from typing import List, Dict, Any
from .base_parser import BaseParser, ParserResult, ParserError


class PDFParser(BaseParser):
    """
    Parser for PDF documents.
    Supports extraction of text, metadata, and basic structure.
    """
    
    def __init__(self):
        """Initialize PDF parser."""
        super().__init__()
        self._fitz = None
    
    def _import_fitz(self):
        """Lazy import of fitz (PyMuPDF)."""
        if self._fitz is None:
            try:
                import fitz
                self._fitz = fitz
            except ImportError:
                raise ParserError(
                    "PyMuPDF not installed. Install with: pip install PyMuPDF"
                )
    
    def can_parse(self, source: str) -> bool:
        """
        Check if source is a PDF file.
        
        Args:
            source: File path or URL.
            
        Returns:
            bool: True if source appears to be a PDF.
        """
        if source.startswith(('http://', 'https://')):
            return source.lower().endswith('.pdf')
        return source.lower().endswith('.pdf') and os.path.exists(source)
    
    def parse(self, source: str) -> ParserResult:
        """
        Parse PDF document.
        
        Args:
            source: Path to PDF file.
            
        Returns:
            ParserResult: Extracted content and metadata.
            
        Raises:
            ParserError: If parsing fails.
        """
        self._import_fitz()
        
        try:
            # Open PDF
            doc = self._fitz.open(source)
            
            # Extract metadata
            metadata = doc.metadata or {}
            title = metadata.get('title', '')
            authors = self._extract_authors(metadata.get('author', ''))
            
            # Extract text from all pages
            full_text = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text:
                    full_text.append(text)
            
            doc.close()
            
            if not full_text:
                raise ParserError("No text extracted from PDF", source)
            
            # Combine and clean text
            raw_content = '\n'.join(full_text)
            content = self._clean_text(raw_content)
            
            # Try to extract abstract
            abstract = self._extract_abstract(raw_content)
            
            result = ParserResult(
                content=content,
                title=title.strip() if title else self._extract_title(raw_content),
                authors=authors,
                abstract=abstract,
                metadata={
                    'source': source,
                    'pages': len(full_text),
                    'pdf_metadata': metadata,
                    'parser_type': 'pdf'
                }
            )
            
            if not self.validate_content(result.content):
                raise ParserError("Extracted content is too short or invalid", source)
            
            return result
            
        except Exception as e:
            if isinstance(e, ParserError):
                raise
            raise ParserError(f"Failed to parse PDF: {str(e)}", source)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw extracted text.
            
        Returns:
            str: Cleaned text.
        """
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove hyphenation at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Normalize spaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _extract_authors(self, author_string: str) -> List[str]:
        """
        Extract authors from metadata string.
        
        Args:
            author_string: Author metadata string.
            
        Returns:
            List[str]: List of author names.
        """
        if not author_string:
            return []
        
        # Split by common separators
        authors = re.split(r'[;,]', author_string)
        return [a.strip() for a in authors if a.strip()]
    
    def _extract_title(self, text: str) -> str:
        """
        Attempt to extract title from text.
        
        Args:
            text: Full document text.
            
        Returns:
            str: Extracted title or empty string.
        """
        lines = text.split('\n')
        
        # Look for title-like patterns in first few lines
        for line in lines[:10]:
            line = line.strip()
            if (len(line) > 20 and len(line) < 200 and 
                not line.lower().startswith(('abstract', 'doi', 'copyright'))):
                return line
        
        return ""
    
    def _extract_abstract(self, text: str) -> str:
        """
        Extract abstract section from text.
        
        Args:
            text: Full document text.
            
        Returns:
            str: Abstract text.
        """
        # Common abstract markers
        patterns = [
            r'(?:ABSTRACT|Аннотация|РЕФЕРАТ)[:\s]*(.*?)(?:\n\s*\n|INTRODUCTION|ВВЕДЕНИЕ|KEYWORDS)',
            r'(?:Abstract)[:\s]*(.*?)(?:\n\s*\n|Introduction|Keywords)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up
                abstract = re.sub(r'\n{3,}', '\n\n', abstract)
                return abstract
        
        return ""
