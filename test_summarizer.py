# test_summarizer.py
from summarizer import summarizer

def test_summarizer():
    print("Тестирование суммаризатора...")
    
    # Тестовый текст
    test_text = """Антикоагулянтная терапия у онкологических больных требует балансирования между рисками тромбозов и кровотечений. 
    Исследование Caravaggio показало эффективность апиксабана у пациентов с рак-ассоциированной венозной тромбоэмболией."""
    
    try:
        # Загружаем модель
        print("Загрузка моделей...")
        summarizer.load_model()
        print("Модели загружены успешно!")
        
        # Тестируем метод
        print("Генерация резюме...")
        summary, key_phrases = summarizer.generate_structured_summary(test_text)
        
        print("УСПЕХ! Резюме сгенерировано:")
        print("=" * 50)
        print(summary)
        print("=" * 50)
        print("Ключевые фразы:", key_phrases)
        
    except Exception as e:
        print(f"ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_summarizer()
