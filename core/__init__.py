# Core module for medical summarizer
"""
Core module containing the main summarization logic and interfaces.
"""

from .summarizer_interface import SummarizerInterface
from .medical_summarizer import MedicalSummarizer

__all__ = ['SummarizerInterface', 'MedicalSummarizer']
