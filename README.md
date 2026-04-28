# Medical Article Summarizer v2.0

Система автоматического резюмирования медицинских научных статей с использованием NLP и поддержкой множественных источников.

## 📋 Описание

Проект представляет собой модульное веб-приложение для автоматического создания структурированных резюме медицинских статей с поддержкой:
- Прямого ввода текста
- PDF документов
- PubMed (PMID, DOI, URL)
- arXiv preprints

## ✨ Новые возможности v2.0

### Модульная архитектура
- **Интерфейсы**: Абстрактные базовые классы для суммаризаторов и парсеров
- **Разделение ответственности**: Отдельные модули для ядра, парсеров и утилит
- **Расширяемость**: Легкое добавление новых парсеров и суммаризаторов

### Парсеры документов
- **PDF Parser**: Извлечение текста из PDF файлов с помощью PyMuPDF
- **PubMed Parser**: Загрузка статей из PubMed через NCBI E-utilities API
- **arXiv Parser**: Получение препринтов из arXiv API

### Улучшенный суммаризатор
- Рефакторинг оригинального кода в класс `MedicalSummarizer`
- Структурированный вывод через `SummaryResult`
- Поддержка как экстрактивной, так и абстрактивной суммаризации

## 🏗️ Архитектура проекта

```
workspace/
├── core/                      # Ядро системы
│   ├── __init__.py
│   ├── summarizer_interface.py  # Абстрактный интерфейс суммаризатора
│   └── medical_summarizer.py    # Реализация медицинского суммаризатора
│
├── parsers/                   # Парсеры документов
│   ├── __init__.py
│   ├── base_parser.py         # Базовый интерфейс парсера
│   ├── pdf_parser.py          # PDF парсер
│   ├── pubmed_parser.py       # PubMed парсер
│   └── arxiv_parser.py        # arXiv парсер
│
├── utils/                     # Утилиты
│   ├── __init__.py
│   └── parser_manager.py      # Менеджер парсеров
│
├── app.py                     # Streamlit приложение
├── summarizer.py              # Оригинальный суммаризатор (для совместимости)
├── requirements.txt           # Зависимости
└── README.md                  # Документация
```

## 🚀 Установка

```bash
# Создание виртуального окружения
conda create -n nlp_env python=3.9
conda activate nlp_env

# Установка зависимостей
pip install -r requirements.txt
```

## 💡 Использование

### Базовое использование (текст)

```python
from core import MedicalSummarizer

summarizer = MedicalSummarizer()
result = summarizer.summarize(text, max_length=400, min_length=150)

print(result.main_summary)
print(result.key_findings)
print(result.key_phrases)
```

### PDF документы

```python
from parsers import PDFParser

parser = PDFParser()
result = parser.parse("path/to/article.pdf")
print(result.title, result.authors)
```

### PubMed статьи

```python
from parsers import PubMedParser

parser = PubMedParser(email="your.email@example.com")
result = parser.parse("12345678")  # PMID
```

### arXiv препринты

```python
from parsers import ArXivParser

parser = ArXivParser()
result = parser.parse("2103.12345")  # arXiv ID
```

## 🔧 Настройки

Для PubMed API рекомендуется получить NCBI API ключ для увеличения лимитов.

## 📝 Технологии

- Python 3.8+
- PyTorch, Transformers
- Streamlit
- NLTK, Sentence Transformers
- PyMuPDF

## 👥 Авторы

Иванченко Александр
