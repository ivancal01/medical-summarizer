# core/summarizer_interface.py
"""
Abstract base class defining the interface for all summarizers.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class SummaryResult:
    """Data class to hold summary results."""
    main_summary: str
    key_findings: List[str]
    key_phrases: List[str]
    practical_insights: List[str]
    medical_concepts: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'main_summary': self.main_summary,
            'key_findings': self.key_findings,
            'key_phrases': self.key_phrases,
            'practical_insights': self.practical_insights,
            'medical_concepts': self.medical_concepts,
            'metadata': self.metadata
        }


class SummarizerInterface(ABC):
    """
    Abstract base class for all summarizer implementations.
    Defines the common interface that all summarizers must implement.
    """
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load necessary models for summarization.
        Should be called before using summarize methods.
        """
        pass
    
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """
        Check if models are loaded and ready.
        
        Returns:
            bool: True if models are loaded, False otherwise.
        """
        pass
    
    @abstractmethod
    def summarize(self, text: str, max_length: int = 400, min_length: int = 150) -> SummaryResult:
        """
        Generate a structured summary from the input text.
        
        Args:
            text: Input text to summarize.
            max_length: Maximum length of the summary.
            min_length: Minimum length of the summary.
            
        Returns:
            SummaryResult: Structured summary with all components.
        """
        pass
    
    @abstractmethod
    def extract_key_phrases(self, text: str, num_phrases: int = 15) -> List[str]:
        """
        Extract key medical phrases from text.
        
        Args:
            text: Input text.
            num_phrases: Number of phrases to extract.
            
        Returns:
            List[str]: List of key medical phrases.
        """
        pass
    
    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess and clean input text.
        
        Args:
            text: Raw input text.
            
        Returns:
            str: Cleaned and preprocessed text.
        """
        pass
