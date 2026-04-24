# parsers/base_parser.py
"""
Base parser interface for all document parsers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ParserResult:
    """Data class to hold parsed document results."""
    content: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'content': self.content,
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'metadata': self.metadata
        }


class BaseParser(ABC):
    """
    Abstract base class for all document parsers.
    Defines the common interface that all parsers must implement.
    """
    
    def __init__(self):
        """Initialize the parser."""
        pass
    
    @abstractmethod
    def parse(self, source: str) -> ParserResult:
        """
        Parse document from source.
        
        Args:
            source: Source of the document (file path, URL, DOI, etc.)
            
        Returns:
            ParserResult: Parsed document content and metadata.
            
        Raises:
            ParserError: If parsing fails.
        """
        pass
    
    @abstractmethod
    def can_parse(self, source: str) -> bool:
        """
        Check if this parser can handle the given source.
        
        Args:
            source: Source to check.
            
        Returns:
            bool: True if parser can handle this source type.
        """
        pass
    
    def validate_content(self, content: str) -> bool:
        """
        Validate parsed content.
        
        Args:
            content: Content to validate.
            
        Returns:
            bool: True if content is valid.
        """
        if not content or not isinstance(content, str):
            return False
        
        # Check minimum length
        if len(content.strip()) < 100:
            return False
        
        return True


class ParserError(Exception):
    """Custom exception for parser errors."""
    
    def __init__(self, message: str, source: str = None):
        self.message = message
        self.source = source
        super().__init__(self.message)
    
    def __str__(self):
        if self.source:
            return f"ParserError for {self.source}: {self.message}"
        return f"ParserError: {self.message}"
