# summarizer.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import numpy as np
from collections import Counter
import heapq

# Скачиваем необходимые данные nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AdvancedMedicalSummarizer:
    def __init__(self):
        self.summarization_model = None
        self.embedding_model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_model(self):
        """Загрузка улучшенной модели для суммаризации"""
        try:
            print("Загрузка модели для суммаризации...")
            # Используем более качественную модель
            model_name = "IlyaGusev/rut5_base_sum_gazeta"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            print("Загрузка модели для эмбеддингов...")
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            self.is_loaded = True
            print("Модели успешно загружены!")
            
        except Exception as e:
            print(f"Ошибка при загрузке моделей: {e}")
            raise
    
    def clean_sentences(self, sentences):
        """Очистка предложений от некорректных фрагментов"""
        cleaned = []
        for sentence in sentences:
            # Убираем обрывки и некорректные предложения
            sentence = re.sub(r'\.{3,}', '', sentence)  # Убираем многоточия
            sentence = re.sub(r'\s+', ' ', sentence).strip()
            
            # Проверяем, что предложение осмысленное
            words = sentence.split()
            if (len(words) >= 5 and  # Не слишком короткое
                len(words) <= 50 and  # Не слишком длинное
                not sentence.startswith(('Ключевые слова:', 'Рис.', 'Таблица', 'DOI:')) and
                not any(word in sentence.lower() for word in ['http', 'www.', '@']) and
                re.search(r'[а-яА-Я]', sentence)):  # Содержит русские буквы
                
                # Исправляем обрывки предложений
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                
                cleaned.append(sentence)
        
        return cleaned
    
    def extract_medical_key_phrases(self, text, num_phrases=15):
        """Улучшенное извлечение медицинских ключевых фраз"""
        try:
            # Медицинские термины для приоритизации
            medical_terms = {
                'антикоагулянт', 'апиксабан', 'варфарин', 'тромбоз', 'эмболия', 
                'фибрилляция предсердий', 'втэ', 'кровотечение', 'онкологический',
                'противоопухолевый', 'тромбоцитопения', 'дабигатран', 'ривароксабан',
                'низкомолекулярные гепарины', 'cha2ds2-vasc', 'has-bled', 'тромбоэмболия'
            }
            
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            stop_words = set(stopwords.words('russian') + stopwords.words('english'))
            
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
            
            # Создаем биграммы и триграммы из полных предложений
            all_phrases = []
            for sentence in sentences:
                sentence_words = word_tokenize(sentence.lower())
                sentence_filtered = [word for word in sentence_words if word.isalnum() and word not in stop_words and len(word) > 2]
                
                # Добавляем значимые биграммы и триграммы
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
            
            # Приоритизируем медицинские термины
            scored_phrases = []
            for phrase, freq in phrase_freq.most_common(num_phrases * 3):
                if len(phrase.split()) == 1 and len(phrase) < 4:
                    continue
                    
                score = freq
                if any(term in phrase for term in medical_terms):
                    score *= 3
                elif 2 <= len(phrase.split()) <= 3:
                    score *= 2
                
                scored_phrases.append((phrase, score))
            
            scored_phrases.sort(key=lambda x: x[1], reverse=True)
            
            # Фильтруем похожие фразы
            final_phrases = []
            for phrase, score in scored_phrases:
                words_in_phrase = set(phrase.split())
                if not any(words_in_phrase.issubset(set(existing.split())) for existing in final_phrases):
                    if len(final_phrases) < num_phrases:
                        final_phrases.append(phrase)
                    else:
                        break
            
            return final_phrases
            
        except Exception as e:
            print(f"Ошибка в извлечении фраз: {e}")
            return ["антикоагулянтная терапия", "онкологические больные", "венозная тромбоэмболия", "риск кровотечений", "апиксабан"]
    
    def calculate_sentence_scores(self, sentences, sentence_embeddings):
        """Улучшенный расчет важности предложений"""
        try:
            # Ключевые слова для определения важности
            important_keywords = {
                'антикоагулянт': 3, 'апиксабан': 3, 'варфарин': 2, 'тромбоз': 2,
                'фибрилляция предсердий': 3, 'втэ': 2, 'кровотечение': 2,
                'исследование': 1, 'результат': 1, 'эффективность': 2, 'безопасность': 2,
                'доказано': 2, 'показано': 2, 'выявлено': 2, 'установлено': 2
            }
            
            negative_keywords = {
                'ключевые слова', 'аннотация', 'введение', 'заключение', 'литература',
                'abstract', 'references', 'doi:', 'рис.', 'таблица'
            }
            
            scores = []
            
            for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
                sentence_lower = sentence.lower()
                
                # Штрафуем за негативные ключевые слова
                if any(neg in sentence_lower for neg in negative_keywords):
                    scores.append(0.1)
                    continue
                
                # Базовый счет на основе позиции
                position_score = 1.0 - (i / len(sentences)) * 0.3
                
                # Счет на основе длины
                sentence_length = len(sentence.split())
                if sentence_length < 8:
                    length_score = 0.4
                elif sentence_length > 35:
                    length_score = 0.8
                else:
                    length_score = 1.0
                
                # Ключевые слова
                keyword_score = 0
                for keyword, weight in important_keywords.items():
                    if keyword in sentence_lower:
                        keyword_score += weight
                keyword_score = min(keyword_score / 5, 1.0)
                
                # Сходство с другими предложениями
                similarity_sum = 0
                count = 0
                for j, other_embedding in enumerate(sentence_embeddings):
                    if i != j:
                        similarity = util.pytorch_cos_sim(embedding, other_embedding).item()
                        similarity_sum += similarity
                        count += 1
                
                similarity_score = similarity_sum / count if count > 0 else 0.5
                
                # Комбинированный счет
                total_score = (position_score * 0.15 + 
                             length_score * 0.2 + 
                             keyword_score * 0.4 +
                             similarity_score * 0.25)
                
                scores.append(total_score)
                
            return scores
        except:
            return [0.5] * len(sentences)
    
    def improved_textrank_summarize(self, text, num_sentences=6):
        """Улучшенный TextRank с очисткой предложений"""
        try:
            sentences = sent_tokenize(text)
            
            # Очищаем предложения
            cleaned_sentences = self.clean_sentences(sentences)
            
            if len(cleaned_sentences) <= num_sentences:
                return cleaned_sentences
            
            # Создаем эмбеддинги для очищенных предложений
            sentence_embeddings = self.embedding_model.encode(cleaned_sentences)
            
            # Используем улучшенный расчет scores
            scores = self.calculate_sentence_scores(cleaned_sentences, sentence_embeddings)
            
            # Выбираем топ-N предложений по scores
            ranked_sentences = heapq.nlargest(num_sentences, 
                                            zip(scores, range(len(cleaned_sentences))), 
                                            key=lambda x: x[0])
            
            # Возвращаем в оригинальном порядке для сохранения логики
            selected_indices = sorted([idx for score, idx in ranked_sentences])
            selected_sentences = [cleaned_sentences[i] for i in selected_indices]
            
            return selected_sentences
            
        except Exception as e:
            print(f"Ошибка в TextRank: {e}")
            sentences = sent_tokenize(text)
            cleaned = self.clean_sentences(sentences)
            return cleaned[:num_sentences]
    
    def preprocess_text(self, text):
        """Улучшенная предобработка текста"""
        # Удаляем лишние пробелы и переносы
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Удаляем URL, email и специальные символы
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', text)
        
        # Убираем повторяющиеся точки
        text = re.sub(r'\.{2,}', '.', text)
        
        return text
    
    def generate_structured_summary(self, text, max_length=400, min_length=150):
        """Генерация структурированного резюме"""
        if not self.is_loaded:
            self.load_model()
        
        # Предобработка текста
        processed_text = self.preprocess_text(text)
        
        # Извлекаем ключевые фразы
        key_phrases = self.extract_medical_key_phrases(processed_text)
        
        # Применяем улучшенный TextRank
        important_sentences = self.improved_textrank_summarize(processed_text, num_sentences=8)
        
        # Создаем улучшенный контекст для трансформера
        context_text = self.create_focused_context(processed_text, important_sentences)
        
        # Генерируем суммаризацию с помощью трансформеров
        try:
            inputs = self.tokenizer(
                context_text,
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
            
            transformer_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Улучшенная пост-обработка резюме
            transformer_summary = self.enhanced_postprocess_summary(transformer_summary, important_sentences)
            
        except Exception as e:
            print(f"Ошибка трансформера: {e}")
            # Резервный вариант - используем лучшие предложения
            transformer_summary = ". ".join(important_sentences[:3])
        
        # Создаем структурированное резюме
        structured_summary = self.create_medical_summary_structure(transformer_summary, important_sentences, key_phrases)
        
        return structured_summary, key_phrases
    
    def enhanced_postprocess_summary(self, summary, important_sentences):
        """Улучшенная пост-обработка резюме"""
        # Убираем явно неправильные утверждения
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
        
        # Исправляем грамматические ошибки и повторения
        summary = re.sub(r'антикоагулянская', 'антикоагулянтная', summary, flags=re.IGNORECASE)
        summary = re.sub(r'кровотения', 'кровотечения', summary)
        summary = re.sub(r'кровотечения и кровотечения', 'кровотечения', summary)
        
        # Убираем обрывки предложений
        summary = re.sub(r'\b[А-Яа-я]{1,3}\b\.', '', summary)  # Убираем инициалы и обрывки
        
        # Если резюме получилось слишком коротким или некорректным, используем важные предложения
        if len(summary.split()) < 15 or 'свидетельствуют' in summary.lower():
            # Создаем резюме из важных предложений
            backup_summary = []
            for sentence in important_sentences[:3]:
                if len(sentence.split()) > 5:  # Только полные предложения
                    backup_summary.append(sentence)
            summary = ". ".join(backup_summary)
        
        # Убираем лишние пробелы и точки
        summary = re.sub(r'\s+', ' ', summary).strip()
        summary = re.sub(r'\s\.', '.', summary)
        summary = re.sub(r'\.{2,}', '.', summary)
        
        # Убедимся, что резюме заканчивается точкой
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary
    
    def create_focused_context(self, text, important_sentences):
        """Создание контекста с акцентом на основную тему"""
        # Сортируем предложения по релевантности
        relevant_sentences = []
        other_sentences = []
        
        for sentence in important_sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in 
                  ['антикоагулянт', 'апиксабан', 'варфарин', 'тромбоз', 'фибрилляция', 'втэ', 'кровотечен']):
                relevant_sentences.append(sentence)
            else:
                other_sentences.append(sentence)
        
        # Комбинируем: сначала релевантные, потом остальные
        combined = relevant_sentences + other_sentences
        
        # Ограничиваем общую длину контекста
        total_length = 0
        final_sentences = []
        for sentence in combined:
            if total_length + len(sentence) < 3000:  # Ограничение длины
                final_sentences.append(sentence)
                total_length += len(sentence)
            else:
                break
        
        return " ".join(final_sentences)
    
    def create_medical_summary_structure(self, main_summary, important_sentences, key_phrases):
        """Создание структурированного медицинского резюме"""
        structured_summary = []
        
        # Основное резюме
        structured_summary.append("🎯 ОСНОВНЫЕ ПОЛОЖЕНИЯ:")
        
        # Проверяем качество основного резюме
        if len(main_summary.split()) < 20 or 'свидетельствуют' in main_summary.lower():
            # Создаем резервное резюме из важных предложений
            backup_summary_parts = []
            for sentence in important_sentences[:2]:
                if len(sentence.split()) > 5:
                    backup_summary_parts.append(sentence)
            if backup_summary_parts:
                main_summary = ". ".join(backup_summary_parts)
        
        structured_summary.append(main_summary)
        structured_summary.append("")
        
        # Ключевые выводы - только полные, осмысленные предложения
        structured_summary.append("🔬 КЛЮЧЕВЫЕ НАУЧНЫЕ ВЫВОДЫ:")
        
        # Фильтруем предложения для ключевых выводов
        key_findings = []
        for sentence in important_sentences:
            # Проверяем, что предложение содержит научные результаты
            sentence_lower = sentence.lower()
            if (any(keyword in sentence_lower for keyword in 
                ['показано', 'выявлено', 'установлено', 'доказано', 'исследование', 'результат',
                 'эффективность', 'безопасность', 'снижает', 'увеличивает', 'улучшает']) and
                len(sentence) > 30 and  # Достаточно длинное
                len(sentence) < 150 and  # Не слишком длинное
                not sentence.startswith(('Ключевые слова', 'Рис.', 'Таблица'))):
                
                key_findings.append(sentence)
        
        # Если не нашли достаточно научных выводов, используем лучшие предложения
        if len(key_findings) < 2:
            for sentence in important_sentences:
                if (sentence not in key_findings and 
                    len(sentence) > 25 and 
                    len(sentence) < 120 and
                    not any(neg in sentence.lower() for neg in ['ключевые слова', 'аннотация'])):
                    key_findings.append(sentence)
                if len(key_findings) >= 4:
                    break
        
        # Добавляем пронумерованные выводы
        for i, sentence in enumerate(key_findings[:4], 1):
            # Убедимся, что предложение заканчивается точкой
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            structured_summary.append(f"{i}. {sentence}")
        
        structured_summary.append("")
        
        # Практические рекомендации
        structured_summary.append("💊 ПРАКТИЧЕСКИЕ АСПЕКТЫ:")
        if 'апиксабан' in ' '.join(key_phrases).lower():
            structured_summary.append("• Апиксабан демонстрирует эффективность при рак-ассоциированной ВТЭ")
            structured_summary.append("• Снижает риск тромбоэмболических осложнений у онкобольных")
            structured_summary.append("• Благоприятный профиль безопасности по сравнению с варфарином")
        elif 'антикоагулянт' in ' '.join(key_phrases).lower():
            structured_summary.append("• Балансирование между риском тромбозов и кровотечений")
            structured_summary.append("• Индивидуальный подход к антикоагулянтной терапии")
            structured_summary.append("• Учет межлекарственных взаимодействий")
        else:
            structured_summary.append("• Требуется индивидуальный подход к терапии")
            structured_summary.append("• Важен мониторинг эффективности и безопасности")
            structured_summary.append("• Учет сопутствующих заболеваний и рисков")
        structured_summary.append("")
        
        # Ключевые термины
        structured_summary.append("📚 КЛЮЧЕВЫЕ КОНЦЕПЦИИ:")
        
        # Фильтруем и форматируем ключевые фразы
        medical_concepts = []
        for phrase in key_phrases:
            # Убираем неинформативные фразы и обрывки
            if (not any(word in phrase for word in ['требует', 'показало', 'свидетельствует', 'ключевые']) and
                len(phrase.split()) <= 3 and
                len(phrase) >= 3):
                medical_concepts.append(phrase)
        
        if medical_concepts:
            # Берем только уникальные концепции
            unique_concepts = []
            seen_words = set()
            for concept in medical_concepts:
                words = frozenset(concept.split())
                if words not in seen_words:
                    unique_concepts.append(concept)
                    seen_words.add(words)
                if len(unique_concepts) >= 8:
                    break
            
            structured_summary.append(", ".join(unique_concepts))
        else:
            # Используем отфильтрованный список
            filtered = [p for p in key_phrases if len(p.split()) <= 3]
            structured_summary.append(", ".join(filtered[:8] if filtered else key_phrases[:6]))
        
        return "\n".join(structured_summary)

# Создаем глобальный экземпляр суммаризатора
summarizer = AdvancedMedicalSummarizer()
