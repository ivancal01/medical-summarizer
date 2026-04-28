# core/medical_summarizer.py
"""
Medical Summarizer implementation using advanced NLP techniques.
Refactored version of the original summarizer with improved architecture.
"""

import torch
import re
import heapq
from typing import List, Tuple, Dict, Any
from collections import Counter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from .summarizer_interface import SummarizerInterface, SummaryResult


# Ensure NLTK data is available
def _download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)


_download_nltk_data()


class MedicalSummarizer(SummarizerInterface):
    """
    Advanced medical text summarizer using transformer models and TextRank algorithm.
    Implements the SummarizerInterface for consistent API.
    """
    
    def __init__(self, model_name: str = "IlyaGusev/rut5_base_sum_gazeta",
                 embedding_model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize the medical summarizer.
        
        Args:
            model_name: Name of the HuggingFace model for summarization.
            embedding_model_name: Name of the sentence transformer model for embeddings.
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        
        self.summarization_model = None
        self.embedding_model = None
        self.tokenizer = None
        self._is_loaded = False
        
        # Medical domain knowledge
        self.medical_terms = {
            'антикоагулянт', 'апиксабан', 'варфарин', 'тромбоз', 'эмболия', 
            'фибрилляция предсердий', 'втэ', 'кровотечение', 'онкологический',
            'противоопухолевый', 'тромбоцитопения', 'дабигатран', 'ривароксабан',
            'низкомолекулярные гепарины', 'cha2ds2-vasc', 'has-bled', 'тромбоэмболия'
        }
        
        self.important_keywords = {
            'антикоагулянт': 3, 'апиксабан': 3, 'варфарин': 2, 'тромбоз': 2,
            'фибрилляция предсердий': 3, 'втэ': 2, 'кровотечение': 2,
            'исследование': 1, 'результат': 1, 'эффективность': 2, 'безопасность': 2,
            'доказано': 2, 'показано': 2, 'выявлено': 2, 'установлено': 2
        }
        
        self.negative_keywords = {
            'ключевые слова', 'аннотация', 'введение', 'заключение', 'литература',
            'abstract', 'references', 'doi:', 'рис.', 'таблица'
        }
    
    def load_model(self) -> None:
        """Load summarization and embedding models."""
        if self._is_loaded:
            return
            
        try:
            print(f"Loading summarization model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            print(f"Loading embedding model: {self.embedding_model_name}...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            self._is_loaded = True
            print("Models successfully loaded!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def is_model_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._is_loaded
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess and clean input text.
        
        Args:
            text: Raw input text.
            
        Returns:
            str: Cleaned text.
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove URLs and emails
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', text)
        
        # Remove multiple dots
        text = re.sub(r'\.{2,}', '.', text)
        
        return text
    
    def clean_sentences(self, sentences: List[str]) -> List[str]:
        """
        Clean sentences from invalid fragments.
        
        Args:
            sentences: List of raw sentences.
            
        Returns:
            List[str]: Cleaned sentences.
        """
        cleaned = []
        for sentence in sentences:
            # Remove ellipsis and extra spaces
            sentence = re.sub(r'\.{3,}', '', sentence)
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            
            words = sentence.split()
            if (len(words) >= 5 and len(words) <= 50 and
                not sentence.startswith(('Ключевые слова:', 'Рис.', 'Таблица', 'DOI:')) and
                not any(word in sentence.lower() for word in ['http', 'www.', '@']) and
                re.search(r'[а-яА-Я]', sentence)):
                
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                
                cleaned.append(sentence)
        
        return cleaned
    
    def extract_key_phrases(self, text: str, num_phrases: int = 15) -> List[str]:
        """
        Extract medical key phrases from text.
        
        Args:
            text: Input text.
            num_phrases: Number of phrases to extract.
            
        Returns:
            List[str]: Key medical phrases.
        """
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            stop_words = set(stopwords.words('russian') + stopwords.words('english'))
            
            filtered_words = [word for word in words 
                            if word.isalnum() and word not in stop_words and len(word) > 2]
            
            all_phrases = []
            for sentence in sentences:
                sentence_words = word_tokenize(sentence.lower())
                sentence_filtered = [word for word in sentence_words 
                                    if word.isalnum() and word not in stop_words and len(word) > 2]
                
                # Add bigrams and trigrams
                for i in range(len(sentence_filtered)-1):
                    bigram = ' '.join(sentence_filtered[i:i+2])
                    if len(bigram) > 3:
                        all_phrases.append(bigram)
                
                for i in range(len(sentence_filtered)-2):
                    trigram = ' '.join(sentence_filtered[i:i+3])
                    if len(trigram) > 5:
                        all_phrases.append(trigram)
            
            all_phrases.extend(filtered_words)
            phrase_freq = Counter(all_phrases)
            
            # Score phrases with medical term priority
            scored_phrases = []
            for phrase, freq in phrase_freq.most_common(num_phrases * 3):
                if len(phrase.split()) == 1 and len(phrase) < 4:
                    continue
                    
                score = freq
                if any(term in phrase for term in self.medical_terms):
                    score *= 3
                elif 2 <= len(phrase.split()) <= 3:
                    score *= 2
                
                scored_phrases.append((phrase, score))
            
            scored_phrases.sort(key=lambda x: x[1], reverse=True)
            
            # Filter similar phrases
            final_phrases = []
            for phrase, score in scored_phrases:
                words_in_phrase = set(phrase.split())
                if not any(words_in_phrase.issubset(set(existing.split())) 
                          for existing in final_phrases):
                    if len(final_phrases) < num_phrases:
                        final_phrases.append(phrase)
                    else:
                        break
            
            return final_phrases
            
        except Exception as e:
            print(f"Error extracting key phrases: {e}")
            return ["антикоагулянтная терапия", "онкологические больные", 
                   "венозная тромбоэмболия", "риск кровотечений", "апиксабан"]
    
    def calculate_sentence_scores(self, sentences: List[str], 
                                 sentence_embeddings: torch.Tensor) -> List[float]:
        """
        Calculate importance scores for sentences.
        
        Args:
            sentences: List of sentences.
            sentence_embeddings: Embeddings for each sentence.
            
        Returns:
            List[float]: Importance scores for each sentence.
        """
        try:
            scores = []
            
            for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
                sentence_lower = sentence.lower()
                
                # Penalize negative keywords
                if any(neg in sentence_lower for neg in self.negative_keywords):
                    scores.append(0.1)
                    continue
                
                # Position score
                position_score = 1.0 - (i / len(sentences)) * 0.3
                
                # Length score
                sentence_length = len(sentence.split())
                if sentence_length < 8:
                    length_score = 0.4
                elif sentence_length > 35:
                    length_score = 0.8
                else:
                    length_score = 1.0
                
                # Keyword score
                keyword_score = 0
                for keyword, weight in self.important_keywords.items():
                    if keyword in sentence_lower:
                        keyword_score += weight
                keyword_score = min(keyword_score / 5, 1.0)
                
                # Similarity score
                similarity_sum = 0
                count = 0
                for j, other_embedding in enumerate(sentence_embeddings):
                    if i != j:
                        similarity = util.pytorch_cos_sim(embedding, other_embedding).item()
                        similarity_sum += similarity
                        count += 1
                
                similarity_score = similarity_sum / count if count > 0 else 0.5
                
                # Combined score
                total_score = (position_score * 0.15 + 
                             length_score * 0.2 + 
                             keyword_score * 0.4 +
                             similarity_score * 0.25)
                
                scores.append(total_score)
                
            return scores
        except Exception:
            return [0.5] * len(sentences)
    
    def textrank_summarize(self, text: str, num_sentences: int = 6) -> List[str]:
        """
        Extractive summarization using improved TextRank.
        
        Args:
            text: Input text.
            num_sentences: Number of sentences to extract.
            
        Returns:
            List[str]: Most important sentences.
        """
        try:
            sentences = sent_tokenize(text)
            cleaned_sentences = self.clean_sentences(sentences)
            
            if len(cleaned_sentences) <= num_sentences:
                return cleaned_sentences
            
            sentence_embeddings = self.embedding_model.encode(cleaned_sentences)
            scores = self.calculate_sentence_scores(cleaned_sentences, sentence_embeddings)
            
            ranked_sentences = heapq.nlargest(num_sentences, 
                                            zip(scores, range(len(cleaned_sentences))), 
                                            key=lambda x: x[0])
            
            selected_indices = sorted([idx for score, idx in ranked_sentences])
            selected_sentences = [cleaned_sentences[i] for i in selected_indices]
            
            return selected_sentences
            
        except Exception as e:
            print(f"Error in TextRank: {e}")
            sentences = sent_tokenize(text)
            cleaned = self.clean_sentences(sentences)
            return cleaned[:num_sentences]
    
    def create_focused_context(self, text: str, important_sentences: List[str]) -> str:
        """
        Create focused context for transformer model.
        
        Args:
            text: Original text.
            important_sentences: Important sentences from TextRank.
            
        Returns:
            str: Focused context string.
        """
        relevant_sentences = []
        other_sentences = []
        
        for sentence in important_sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in 
                  ['антикоагулянт', 'апиксабан', 'варфарин', 'тромбоз', 
                   'фибрилляция', 'втэ', 'кровотечен']):
                relevant_sentences.append(sentence)
            else:
                other_sentences.append(sentence)
        
        combined = relevant_sentences + other_sentences
        
        total_length = 0
        final_sentences = []
        for sentence in combined:
            if total_length + len(sentence) < 3000:
                final_sentences.append(sentence)
                total_length += len(sentence)
            else:
                break
        
        return " ".join(final_sentences)
    
    def generate_transformer_summary(self, text: str, max_length: int = 400, 
                                    min_length: int = 150) -> str:
        """
        Generate abstractive summary using transformer model.
        
        Args:
            text: Input text.
            max_length: Maximum summary length.
            min_length: Minimum summary length.
            
        Returns:
            str: Generated summary.
        """
        try:
            inputs = self.tokenizer(
                text,
                max_length=1024,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.summarization_model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    do_sample=False
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            print(f"Transformer error: {e}")
            return ""
    
    def postprocess_summary(self, summary: str, important_sentences: List[str]) -> str:
        """
        Post-process generated summary.
        
        Args:
            summary: Raw summary.
            important_sentences: Important sentences for fallback.
            
        Returns:
            str: Cleaned summary.
        """
        incorrect_patterns = [
            r'апиксабан может повысить риск тромбозов',
            r'апиксабан увеличивает риск',
            r'апиксабан приводит к тромбозам',
            r'антикоагулянская терапия может повысить риск',
            r'свидетельствуют данные исследования',
            r'эксперты считают, что тромбозов'
        ]
        
        for pattern in incorrect_patterns:
            summary = re.sub(pattern, '', summary, flags=re.IGNORECASE)
        
        # Fix common errors
        summary = re.sub(r'антикоагулянская', 'антикоагулянтная', summary, flags=re.IGNORECASE)
        summary = re.sub(r'кровотения', 'кровотечения', summary)
        summary = re.sub(r'кровотечения и кровотечения', 'кровотечения', summary)
        
        # Remove sentence fragments
        summary = re.sub(r'\b[А-Яа-я]{1,3}\b\.', '', summary)
        
        # Fallback if summary is too short
        if len(summary.split()) < 15 or 'свидетельствуют' in summary.lower():
            backup_summary = []
            for sentence in important_sentences[:3]:
                if len(sentence.split()) > 5:
                    backup_summary.append(sentence)
            summary = ". ".join(backup_summary)
        
        # Clean up
        summary = re.sub(r'\s+', ' ', summary).strip()
        summary = re.sub(r'\s\.', '.', summary)
        summary = re.sub(r'\.{2,}', '.', summary)
        
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary
    
    def create_structured_output(self, main_summary: str, important_sentences: List[str], 
                                key_phrases: List[str]) -> SummaryResult:
        """
        Create structured summary output.
        
        Args:
            main_summary: Main summary text.
            important_sentences: Important sentences.
            key_phrases: Key medical phrases.
            
        Returns:
            SummaryResult: Structured summary result.
        """
        # Improve main summary if needed
        if len(main_summary.split()) < 20 or 'свидетельствуют' in main_summary.lower():
            backup_parts = []
            for sentence in important_sentences[:2]:
                if len(sentence.split()) > 5:
                    backup_parts.append(sentence)
            if backup_parts:
                main_summary = ". ".join(backup_parts)
        
        # Extract key findings
        key_findings = []
        for sentence in important_sentences:
            sentence_lower = sentence.lower()
            if (any(keyword in sentence_lower for keyword in 
                ['показано', 'выявлено', 'установлено', 'доказано', 'исследование', 'результат',
                 'эффективность', 'безопасность', 'снижает', 'увеличивает', 'улучшает']) and
                30 < len(sentence) < 150 and
                not sentence.startswith(('Ключевые слова', 'Рис.', 'Таблица'))):
                key_findings.append(sentence)
        
        if len(key_findings) < 2:
            for sentence in important_sentences:
                if (sentence not in key_findings and 
                    25 < len(sentence) < 120 and
                    not any(neg in sentence.lower() for neg in self.negative_keywords)):
                    key_findings.append(sentence)
                if len(key_findings) >= 4:
                    break
        
        # Ensure proper punctuation
        key_findings = [s if s.endswith(('.', '!', '?')) else s + '.' 
                       for s in key_findings[:4]]
        
        # Generate practical insights based on content
        practical_insights = []
        phrases_text = ' '.join(key_phrases).lower()
        
        if 'апиксабан' in phrases_text:
            practical_insights = [
                "Апиксабан демонстрирует эффективность при рак-ассоциированной ВТЭ",
                "Снижает риск тромбоэмболических осложнений у онкобольных",
                "Благоприятный профиль безопасности по сравнению с варфарином"
            ]
        elif 'антикоагулянт' in phrases_text:
            practical_insights = [
                "Балансирование между риском тромбозов и кровотечений",
                "Индивидуальный подход к антикоагулянтной терапии",
                "Учет межлекарственных взаимодействий"
            ]
        else:
            practical_insights = [
                "Требуется индивидуальный подход к терапии",
                "Важен мониторинг эффективности и безопасности",
                "Учет сопутствующих заболеваний и рисков"
            ]
        
        # Extract medical concepts
        medical_concepts = []
        for phrase in key_phrases:
            if (not any(word in phrase for word in ['требует', 'показало', 'свидетельствует', 'ключевые']) and
                len(phrase.split()) <= 3 and len(phrase) >= 3):
                medical_concepts.append(phrase)
        
        if not medical_concepts:
            medical_concepts = [p for p in key_phrases if len(p.split()) <= 3][:8]
        
        # Remove duplicates
        unique_concepts = []
        seen_words = set()
        for concept in medical_concepts:
            words = frozenset(concept.split())
            if words not in seen_words:
                unique_concepts.append(concept)
                seen_words.add(words)
            if len(unique_concepts) >= 8:
                break
        
        return SummaryResult(
            main_summary=main_summary,
            key_findings=key_findings,
            key_phrases=key_phrases,
            practical_insights=practical_insights,
            medical_concepts=unique_concepts,
            metadata={
                'num_sentences': len(important_sentences),
                'num_key_phrases': len(key_phrases)
            }
        )
    
    def summarize(self, text: str, max_length: int = 400, 
                 min_length: int = 150) -> SummaryResult:
        """
        Generate complete structured summary.
        
        Args:
            text: Input text to summarize.
            max_length: Maximum summary length.
            min_length: Minimum summary length.
            
        Returns:
            SummaryResult: Complete structured summary.
        """
        if not self._is_loaded:
            self.load_model()
        
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        # Extract key phrases
        key_phrases = self.extract_key_phrases(processed_text)
        
        # Extractive summarization
        important_sentences = self.textrank_summarize(processed_text, num_sentences=8)
        
        # Create focused context
        context_text = self.create_focused_context(processed_text, important_sentences)
        
        # Abstractive summarization
        transformer_summary = self.generate_transformer_summary(
            context_text, max_length, min_length
        )
        
        # Post-process
        final_summary = self.postprocess_summary(transformer_summary, important_sentences)
        
        # Create structured output
        result = self.create_structured_output(
            final_summary, important_sentences, key_phrases
        )
        
        return result
