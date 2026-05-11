# core/medical_summarizer.py
"""
Universal Medical Summarizer with Auto-Strategy Selection.
Automatically detects model type (Generative vs Extractive) and applies the best strategy.
"""

import torch
import re
import heapq
from typing import List, Dict, Any, Optional
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from sentence_transformers import SentenceTransformer, util
import nltk

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
except ImportError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize

from .summarizer_interface import SummarizerInterface, SummaryResult

# Ensure NLTK data
def _download_nltk_data():
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt', quiet=True)
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords', quiet=True)

_download_nltk_data()

class MedicalSummarizer(SummarizerInterface):
    """
    Universal Medical Summarizer.
    - Uses Generative strategy for T5, BART, RuT5 models.
    - Uses Extractive strategy (Sentence Embeddings) for BERT, RoBERTa, RuBioRoBERTa models.
    """

    def __init__(self, model_name: str = "IlyaGusev/rut5_base_sum_gazeta"):
        self.model_name = model_name
        self.summarization_model = None
        self.tokenizer = None
        self.embedding_model = None
        
        # Strategy flags
        self.is_generative = False
        self.is_extractive = False
        self._is_loaded = False

        # Domain knowledge (kept from original)
        self.medical_terms = {
            'антикоагулянт', 'апиксабан', 'варфарин', 'тромбоз', 'эмболия',
            'фибрилляция предсердий', 'втэ', 'кровотечение', 'онкологический',
            'противоопухолевый', 'тромбоцитопения', 'дабигатран', 'ривароксабан',
            'низкомолекулярные гепарины', 'cha2ds2-vasc', 'has-bled', 'тромбоэмболия'
        }
        self.important_keywords = {
            'антикоагулянт': 3, 'апиксабан': 3, 'варфарин': 2, 'тромбоз': 2,
            'фибрилляция предсердий': 3, 'втэ': 2, 'кровотечение': 2,
            'исследование': 1, 'результат': 1, 'эффективность': 2, 'безопасность': 2
        }
        self.negative_keywords = {
            'ключевые слова', 'аннотация', 'введение', 'заключение', 'литература',
            'abstract', 'references', 'doi:', 'рис.', 'таблица'
        }

    def _detect_strategy(self) -> None:
        """Detect strategy based on model name."""
        name_lower = self.model_name.lower()
        
        # Generative models (Seq2Seq)
        generative_keywords = ['t5', 'bart', 'mt5', 'byt5', 'pegasus']
        # Extractive models (Encoder-only)
        extractive_keywords = ['bert', 'roberta', 'rubio', 'deberta', 'electra', 'sentence-transformers']

        if any(k in name_lower for k in generative_keywords):
            self.is_generative = True
            self.is_extractive = False
            print(f"🤖 Detected GENERATIVE strategy for: {self.model_name}")
        elif any(k in name_lower for k in extractive_keywords):
            self.is_generative = False
            self.is_extractive = True
            print(f"🔍 Detected EXTRACTIVE strategy for: {self.model_name}")
        else:
            # Default to generative if unsure, but warn
            print(f"⚠️ Unknown model type for {self.model_name}. Defaulting to GENERATIVE.")
            self.is_generative = True
            self.is_extractive = False

    def load_model(self) -> None:
        """Load models based on detected strategy."""
        if self._is_loaded:
            return

        self._detect_strategy()

        try:
            if self.is_generative:
                print(f"Loading Generative Model: {self.model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            elif self.is_extractive:
                print(f"Loading Extractive Model: {self.model_name}...")
                # For extractive, we use the model as a sentence encoder
                # If it's a standard transformer, wrap it or use SentenceTransformer if compatible
                try:
                    # Try loading as SentenceTransformer first (optimal for embeddings)
                    self.embedding_model = SentenceTransformer(self.model_name)
                except Exception:
                    # Fallback: Load as standard transformer and pool manually
                    print("Standard transformer detected, using pooling for embeddings...")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.summarization_model = AutoModel.from_pretrained(self.model_name)
                    self.embedding_model = None # Will handle manually

            self._is_loaded = True
            print("✅ Models loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise

    def is_model_loaded(self) -> bool:
        return self._is_loaded

    def _get_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """Get embeddings for sentences."""
        if self.embedding_model:
            return self.embedding_model.encode(sentences, convert_to_tensor=True)
        
        # Manual pooling if using standard transformer
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.summarization_model(**inputs)
        
        # Mean pooling
        embeddings = []
        for i in range(len(sentences)):
            token_embeddings = outputs.last_hidden_state[i]
            attention_mask = inputs['attention_mask'][i]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0)
            sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9)
            embeddings.append(sum_embeddings / sum_mask)
        
        return torch.stack(embeddings)

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', text)
        return text

    def clean_sentences(self, sentences: List[str]) -> List[str]:
        cleaned = []
        for s in sentences:
            s = re.sub(r'\.{3,}', '', s).strip()
            words = s.split()
            if (len(words) >= 5 and len(words) <= 60 and 
                not any(neg in s.lower() for neg in self.negative_keywords) and
                re.search(r'[а-яА-Яa-zA-Z]', s)):
                if not s.endswith(('.', '!', '?')): s += '.'
                cleaned.append(s)
        return cleaned

    def _extractive_summary(self, text: str, num_sentences: int = 5) -> str:
        """Perform extractive summarization using embeddings."""
        sentences = sent_tokenize(text)
        clean_sents = self.clean_sentences(sentences)
        
        if len(clean_sents) <= num_sentences:
            return " ".join(clean_sents)

        try:
            embeddings = self._get_embeddings(clean_sents)
            
            # Calculate centroid
            centroid = torch.mean(embeddings, dim=0)
            
            # Calculate similarity to centroid
            scores = []
            for i, emb in enumerate(embeddings):
                sim = util.pytorch_cos_sim(emb.unsqueeze(0), centroid.unsqueeze(0))[0][0].item()
                
                # Boost score for medical keywords
                boost = 0
                s_lower = clean_sents[i].lower()
                for kw, weight in self.important_keywords.items():
                    if kw in s_lower: boost += weight * 0.1
                
                # Penalize short/long sentences slightly
                length_factor = 1.0 if 10 < len(clean_sents[i].split()) < 40 else 0.8
                
                final_score = (sim * 0.7) + (boost * 0.2) + (length_factor * 0.1)
                scores.append((final_score, i))
            
            # Select top sentences
            top_indices = sorted([idx for _, idx in heapq.nlargest(num_sentences, scores)])
            result_sents = [clean_sents[i] for i in top_indices]
            
            return " ".join(result_sents)
            
        except Exception as e:
            print(f"Extraction error: {e}. Fallback to first sentences.")
            return " ".join(clean_sents[:num_sentences])

    def _generative_summary(self, text: str, max_length: int = 400, min_length: int = 150) -> str:
        """Perform generative summarization."""
        try:
            inputs = self.tokenizer(
                text, max_length=1024, padding=True, truncation=True, return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.summarization_model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Generation error: {e}. Falling back to extractive.")
            return self._extractive_summary(text, num_sentences=5)

    def summarize(self, text: str, max_length: int = 400, min_length: int = 150) -> SummaryResult:
        """Main summarization method with auto-strategy."""
        if not self._is_loaded:
            self.load_model()

        processed_text = self.preprocess_text(text)
        
        # Choose strategy
        if self.is_extractive:
            summary_text = self._extractive_summary(processed_text, num_sentences=5)
            # Generate simple metadata for extractive
            key_phrases = [processed_text.split()[i:i+2] for i in range(0, len(processed_text.split()), 10)]
            key_phrases = [" ".join(p) for p in key_phrases if len(p) > 1][:10]
        else:
            summary_text = self._generative_summary(processed_text, max_length, min_length)
            # Fallback if generation fails or returns garbage
            if len(summary_text.split()) < 10:
                summary_text = self._extractive_summary(processed_text, num_sentences=5)
            key_phrases = [] # Could add extraction here too

        # Create Result Object
        # Adapting to the expected SummaryResult structure
        return SummaryResult(
            main_summary=summary_text,
            key_findings=[summary_text], # Simplified for now
            key_phrases=key_phrases,
            practical_insights=[],
            medical_concepts=[],
            metadata={
                'strategy': 'generative' if self.is_generative else 'extractive',
                'model': self.model_name
            }
        )
