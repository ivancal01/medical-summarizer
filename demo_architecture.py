#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script showing the new modular architecture usage.
This script demonstrates how to use the refactored summarizer and parsers.
"""

def demo_summarizer_interface():
    """Demonstrate the new summarizer interface."""
    print("=" * 60)
    print("DEMO 1: Medical Summarizer with New Interface")
    print("=" * 60)
    
    from core import MedicalSummarizer
    
    # Initialize summarizer
    summarizer = MedicalSummarizer()
    
    # Sample medical text
    test_text = """
    Антикоагулянтная терапия в условиях особых рисков тромбозов и кровотечений 
    у онкологических больных. Количество онкологических больных, которым необходимы 
    консультации и/или наблюдение кардиолога, всё возрастает. Высокую угрозу 
    летальности представляют возникающие у онкологических больных тромбозы и эмболии. 
    Важной причиной повышенной частоты эмболий при онкологических заболеваниях является 
    фибрилляция предсердий. Венозная тромбоэмболия, включая повторные тромбоэмболические 
    события на фоне проводимой антикоагулянтной терапии, служит частой причиной смерти 
    у онкологических больных. Исследования показали, что апиксабан демонстрирует 
    эффективность и безопасность при лечении рак-ассоциированной венозной тромбоэмболии.
    """
    
    print("\nInput text length:", len(test_text), "characters")
    print("\nGenerating summary...")
    
    # Generate summary (models will be loaded on first call)
    result = summarizer.summarize(test_text, max_length=200, min_length=100)
    
    print("\n--- SUMMARY RESULT ---")
    print(f"\n📝 Main Summary:\n{result.main_summary}")
    print(f"\n🔬 Key Findings ({len(result.key_findings)}):")
    for i, finding in enumerate(result.key_findings, 1):
        print(f"  {i}. {finding}")
    print(f"\n🔑 Key Phrases ({len(result.key_phrases)}):")
    print("  ", ", ".join(result.key_phrases[:5]))
    print(f"\n💊 Practical Insights:")
    for insight in result.practical_insights:
        print(f"  • {insight}")
    print(f"\n📚 Medical Concepts:")
    print("  ", ", ".join(result.medical_concepts[:5]))
    
    return result


def demo_parser_interfaces():
    """Demonstrate parser interfaces without actually calling APIs."""
    print("\n\n" + "=" * 60)
    print("DEMO 2: Parser Interfaces")
    print("=" * 60)
    
    from parsers import PDFParser, PubMedParser, ArXivParser
    
    # PDF Parser
    print("\n📄 PDF Parser:")
    pdf_parser = PDFParser()
    print(f"  Can parse 'article.pdf': {pdf_parser.can_parse('article.pdf')}")
    print(f"  Can parse 'doc.txt': {pdf_parser.can_parse('doc.txt')}")
    print(f"  Supported: .pdf files")
    
    # PubMed Parser
    print("\n🏥 PubMed Parser:")
    pubmed_parser = PubMedParser(email="demo@example.com")
    test_sources = [
        "12345678",  # PMID
        "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        "10.1056/NEJMoa1215988",  # DOI
        "pmid:87654321"
    ]
    for source in test_sources:
        can_parse = pubmed_parser.can_parse(source)
        print(f"  Can parse '{source}': {can_parse}")
    
    # arXiv Parser
    print("\n📚 arXiv Parser:")
    arxiv_parser = ArXivParser()
    arxiv_sources = [
        "2103.12345",
        "https://arxiv.org/abs/2103.12345",
        "arxiv:2201.01234",
        "hep-th/9901001"
    ]
    for source in arxiv_sources:
        can_parse = arxiv_parser.can_parse(source)
        print(f"  Can parse '{source}': {can_parse}")


def demo_parser_manager():
    """Demonstrate the parser manager."""
    print("\n\n" + "=" * 60)
    print("DEMO 3: Parser Manager (Automatic Selection)")
    print("=" * 60)
    
    from utils import ParserManager
    
    manager = ParserManager(pubmed_email="demo@example.com")
    
    print("\nSupported formats:")
    formats = manager.get_supported_formats()
    for fmt, desc in formats.items():
        print(f"  {fmt}: {desc}")
    
    print("\nAutomatic parser selection:")
    test_sources = [
        "document.pdf",
        "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        "2103.12345",
        "https://arxiv.org/abs/2201.01234"
    ]
    
    for source in test_sources:
        parser = manager.get_parser(source)
        parser_name = parser.__class__.__name__ if parser else "None"
        print(f"  '{source}' → {parser_name}")


def demo_data_classes():
    """Demonstrate the data classes."""
    print("\n\n" + "=" * 60)
    print("DEMO 4: Data Classes")
    print("=" * 60)
    
    from core.summarizer_interface import SummaryResult
    from parsers.base_parser import ParserResult
    
    # SummaryResult
    print("\n📋 SummaryResult structure:")
    sample_summary = SummaryResult(
        main_summary="Example summary text...",
        key_findings=["Finding 1", "Finding 2"],
        key_phrases=["term1", "term2"],
        practical_insights=["Insight 1"],
        medical_concepts=["concept1"],
        metadata={"num_sentences": 5}
    )
    print(f"  Fields: main_summary, key_findings, key_phrases,")
    print(f"          practical_insights, medical_concepts, metadata")
    print(f"  Convert to dict: {type(sample_summary.to_dict())}")
    
    # ParserResult
    print("\n📄 ParserResult structure:")
    sample_doc = ParserResult(
        content="Document content...",
        title="Example Paper",
        authors=["Author One", "Author Two"],
        abstract="Paper abstract...",
        metadata={"source": "test", "pages": 10}
    )
    print(f"  Fields: content, title, authors, abstract, metadata")
    print(f"  Authors: {sample_doc.authors}")
    print(f"  Metadata keys: {list(sample_doc.metadata.keys())}")


def show_architecture_benefits():
    """Show benefits of the new architecture."""
    print("\n\n" + "=" * 60)
    print("ARCHITECTURE BENEFITS")
    print("=" * 60)
    
    benefits = """
    ✅ Modular Design:
       - Separate modules for core logic, parsers, and utilities
       - Easy to maintain and extend
    
    ✅ Abstract Interfaces:
       - SummarizerInterface defines contract for all summarizers
       - BaseParser provides common parser interface
       - Easy to add new implementations
    
    ✅ Type Safety:
       - Data classes for structured results
       - Type hints throughout the codebase
       - Better IDE support and error detection
    
    ✅ Separation of Concerns:
       - Parsing logic separate from summarization
       - Each parser handles one format
       - Clean API between components
    
    ✅ Testability:
       - Components can be tested independently
       - Mock interfaces for unit testing
       - Demo script shows usage patterns
    
    ✅ Extensibility:
       - Add new parsers by extending BaseParser
       - Create custom summarizers implementing SummarizerInterface
       - ParserManager supports dynamic parser registration
    """
    print(benefits)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MEDICAL SUMMARIZER v2.0 - ARCHITECTURE DEMO")
    print("=" * 60)
    
    try:
        # Note: Full summarizer demo requires model download
        # Uncomment to run full demo:
        # demo_summarizer_interface()
        
        demo_parser_interfaces()
        demo_parser_manager()
        demo_data_classes()
        show_architecture_benefits()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("=" * 60)
        print("\nTo run full summarizer demo, ensure you have:")
        print("  - Internet connection for model download")
        print("  - Sufficient disk space (~2GB for models)")
        print("  - Run: pip install -r requirements.txt")
        print("\nThen uncomment demo_summarizer_interface() in this script.")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
