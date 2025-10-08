# Medical Article Summarizer

Система автоматического резюмирования медицинских научных статей с использованием NLP

## Описание
Проект представляет собой веб-приложение для автоматического создания структурированных резюме медицинских статей на русском языке.

## Функциональность
- 📝 Суммаризация медицинских текстов
- 🔍 Извлечение ключевых терминов  
- 🎯 Структурированные резюме
- ⚡ Быстрая обработка

## Технологии
- Python 3.8+
- PyTorch, Transformers
- Streamlit
- NLTK, Sentence Transformers

## Установка и запуск

conda create -n nlp_env python=3.8
conda activate nlp_env
pip install -r requirements.txt
streamlit run app.py

## Структура проекта
- app.py - основной файл Streamlit приложения
- summarizer.py - логика суммаризации статей
- requirements.txt - зависимости проекта

## Автор
Иванченко Александр
