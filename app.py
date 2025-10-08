# app.py
import streamlit as st
import time
from summarizer import summarizer

# Настройка страницы
st.set_page_config(
    page_title="Advanced Medical Article Summarizer",
    page_icon="🏥",
    layout="wide"
)

# CSS для улучшения внешнего вида
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 20px;
        white-space: pre-line;
    }
    .key-phrases-box {
        background-color: #e7f3ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin-top: 15px;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Заголовок приложения
    st.markdown('<h1 class="main-header">🏥 Advanced Medical Article Summarizer</h1>', unsafe_allow_html=True)
    
    # Описание
    st.write("""
    ### Улучшенный инструмент для суммаризации медицинских статей
    Создает детальные, структурированные резюме с ключевыми выводами и терминами.
    """)
    
    # Инициализация session_state
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        st.subheader("Длина резюме")
        max_length = st.slider("Максимальная длина", 150, 500, 400)
        min_length = st.slider("Минимальная длина", 50, 200, 150)
        
        st.markdown("---")
        st.header("ℹ️ О улучшениях")
        st.info("""
        **Новые возможности:**
        - Тематический анализ
        - Приоритизация медицинских терминов
        - Структурированное резюме
        - Практические рекомендации
        """)
    
    # Основная область ввода
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Входной текст медицинской статьи")
        input_text = st.text_area(
            "Вставьте текст статьи:",
            height=400,
            placeholder="Введите или вставьте текст медицинской статьи здесь...",
            help="Для лучших результатов используйте статьи на русском языке объемом от 1000 символов",
            value=st.session_state.input_text,
            key="text_input"
        )
    
    with col2:
        st.subheader("💡 Рекомендации")
        st.success("""
        **Для лучших результатов:**
        - Полные абзацы текста
        - Структурированный материал
        - Медицинская терминология
        - Четкие формулировки
        """)
        
        # Пример текста для тестирования
        with st.expander("🧪 Тестовый пример"):
            test_text = """Антикоагулянтная терапия в условиях особых рисков тромбозов и кровотечений у онкологических больных. Количество онкологических больных, которым необходимы консультации и/или наблюдение кардиолога, всё возрастает. Высокую угрозу летальности представляют возникающие у онкологических больных тромбозы и эмболии. Важной причиной повышенной частоты эмболий при онкологических заболеваниях является фибрилляция предсердий. Венозная тромбоэмболия, включая повторные тромбоэмболические события на фоне проводимой антикоагулянтной терапии, служит частой причиной смерти у онкологических больных."""
            
            if st.button("Использовать пример"):
                st.session_state.input_text = test_text
                st.rerun()
    
    # Кнопка генерации
    if st.button("🎯 Сгенерировать развернутое резюме", type="primary", use_container_width=True):
        if not input_text:
            st.error("⚠️ Пожалуйста, введите текст статьи")
        elif len(input_text) < 200:
            st.error("⚠️ Текст слишком короткий. Для качественного анализа需要 текст от 500 символов.")
        else:
            with st.spinner("🔄 Анализируем текст и генерируем детальное резюме..."):
                try:
                    # Загрузка модели при первом использовании
                    if not summarizer.is_loaded:
                        summarizer.load_model()
                    
                    # Генерация улучшенного резюме - ИСПРАВЛЕННЫЙ ВЫЗОВ МЕТОДА
                    start_time = time.time()
                    summary, key_phrases = summarizer.generate_structured_summary(
                        input_text, max_length, min_length
                    )
                    processing_time = time.time() - start_time
                    
                    # Вывод результата
                    st.markdown("---")
                    st.markdown('<h2 class="section-header">📋 Результат суммаризации</h2>', unsafe_allow_html=True)
                    
                    # Статистика
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Исходный текст", f"{len(input_text):,} симв.")
                    with col2:
                        st.metric("Резюме", f"{len(summary):,} симв.")
                    with col3:
                        compression_ratio = len(summary)/len(input_text)*100 if len(input_text) > 0 else 0
                        st.metric("Коэффициент сжатия", f"{compression_ratio:.1f}%")
                    with col4:
                        st.metric("Время обработки", f"{processing_time:.2f} сек.")
                    
                    # Детальное резюме
                    st.markdown("### 🎯 Структурированное резюме")
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.write(summary)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Ключевые фразы
                    st.markdown("### 🔍 Ключевые термины и фразы")
                    st.markdown('<div class="key-phrases-box">', unsafe_allow_html=True)
                    
                    # Отображаем ключевые фразы в виде тегов
                    cols = st.columns(4)
                    for i, phrase in enumerate(key_phrases[:12]):
                        with cols[i % 4]:
                            st.markdown(f"`{phrase}`")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Кнопка копирования
                    st.markdown("### 📋 Код для копирования")
                    st.code(summary, language=None)
                    
                except Exception as e:
                    st.error(f"❌ Произошла ошибка при обработке: {str(e)}")
                    st.info("""
                    **Возможные решения:**
                    - Проверьте подключение к интернету
                    - Убедитесь, что установлены все зависимости
                    - Попробуйте уменьшить длину текста
                    - Перезапустите приложение
                    """)

if __name__ == "__main__":
    main()
