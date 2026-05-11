import streamlit as st
import os
import sys
from pathlib import Path

# Настройка путей для импорта
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Импорт компонентов
try:
    from parsers.parser_manager import ParserManager
    from core.medical_summarizer import MedicalSummarizer
except ImportError as e:
    st.error(f"Ошибка импорта: {e}. Убедитесь, что папки parsers и core существуют.")
    st.stop()

# Конфигурация страницы
st.set_page_config(
    page_title="Medical Summarizer Pro",
    page_icon="🩺",
    layout="wide"
)

# CSS стили
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .result-box { background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# Инициализация (кэширование)
@st.cache_resource
def load_system():
    parser_manager = ParserManager()
    summarizer = MedicalSummarizer()
    # Модель загружается лениво внутри суммаризатора при первом вызове
    return parser_manager, summarizer

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    source_type = st.radio(
        "Источник данных:",
        ["📄 Загрузить PDF", "🔍 PubMed (ID/URL)", "📚 arXiv (ID/URL)", "📝 Вставить текст"]
    )
    
    st.divider()
    
    max_length = st.slider("Макс. длина саммари (слов):", 50, 500, 200, 50)
    min_length = st.slider("Мин. длина саммари (слов):", 10, 100, 50, 10)
    
    if st.button("🔄 Сбросить"):
        st.rerun()

# Основная часть
st.markdown('<p class="main-header">🩺 Medical Summarizer Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Анализ медицинских статей: PDF, PubMed, arXiv</p>', unsafe_allow_html=True)

# Загрузка системы
with st.spinner("Инициализация системы..."):
    try:
        parser_manager, summarizer = load_system()
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        st.stop()

# Ввод данных
input_data = None
col1, col2 = st.columns([3, 1])

with col1:
    if source_type == "📄 Загрузить PDF":
        uploaded_file = st.file_uploader("Загрузите PDF", type=["pdf"])
        if uploaded_file:
            temp_path = "temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            input_data = temp_path
            
    elif source_type == "🔍 PubMed (ID/URL)":
        pm_input = st.text_input("PMID или URL:", placeholder="12345678 или https://...")
        if pm_input:
            input_data = pm_input.strip()
            
    elif source_type == "📚 arXiv (ID/URL)":
        ax_input = st.text_input("arXiv ID или URL:", placeholder="2103.12345 или https://...")
        if ax_input:
            input_data = ax_input.strip()
            
    elif source_type == "📝 Вставить текст":
        txt_input = st.text_area("Текст статьи:", height=200)
        if txt_input:
            input_data = txt_input

with col2:
    st.markdown("### Действия")
    run_btn = st.button("🚀 Анализировать", type="primary", disabled=(input_data is None))
    if input_data:
        st.success("Готово к работе")

# Логика обработки
if run_btn and input_data:
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # 1. Парсинг
        status.text("📥 Извлечение текста...")
        if source_type == "📝 Вставить текст":
            # Для текста создаем фейковый результат
            from core.models import ParserResult # Если есть такой класс, иначе используем простой dict
            # Простая эмуляция, если ParserResult нет в этом репозитории
            class FakeResult:
                def __init__(self, text): self.text = text; self.metadata = {"source": "manual"}
            parser_result = FakeResult(input_data)
        else:
            parser_result = parser_manager.parse(input_data)
        
        if not parser_result or not getattr(parser_result, 'text', None):
            st.error("Не удалось извлечь текст.")
            st.stop()
            
        progress.progress(33)
        
        # 2. Суммаризация
        status.text("🧠 Генерация саммари...")
        # Вызов метода summarize. Возвращает объект SummaryResult или строку
        summary_obj = summarizer.summarize(
            parser_result.text, 
            max_length=max_length, 
            min_length=min_length
        )
        
        # Извлечение текста из результата
        if hasattr(summary_obj, 'main_summary'):
            final_text = summary_obj.main_summary
            key_findings = getattr(summary_obj, 'key_findings', [])
            phrases = getattr(summary_obj, 'key_phrases', [])
        elif isinstance(summary_obj, str):
            final_text = summary_obj
            key_findings = []
            phrases = []
        else:
            final_text = str(summary_obj)
            key_findings = []
            phrases = []
            
        progress.progress(100)
        status.text("✅ Готово!")
        
        # 3. Вывод результатов
        st.divider()
        st.subheader("📊 Результаты")
        
        # Метаданные
        if hasattr(parser_result, 'metadata') and parser_result.metadata:
            with st.expander("📋 Метаданные", expanded=True):
                for k, v in parser_result.metadata.items():
                    st.write(f"**{k}:** {v}")
        
        # Саммари
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("### 📝 Краткое содержание")
        st.write(final_text)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Ключевые находки
        if key_findings:
            with st.expander("💡 Ключевые выводы"):
                for item in key_findings:
                    st.write(f"- {item}")
        
        # Фразы
        if phrases:
            with st.expander("🏷️ Ключевые термины"):
                st.write(", ".join(phrases[:15]))
                
        # Статистика
        c1, c2 = st.columns(2)
        orig_len = len(parser_result.text.split())
        summ_len = len(final_text.split())
        c1.metric("Оригинал (слов)", orig_len)
        c2.metric("Саммари (слов)", summ_len)
        
        # Очистка
        if source_type == "📄 Загрузить PDF" and os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
            
    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")
        st.exception(e)
        progress.empty()

st.markdown("---")
st.caption("Powered by Medical Summarizer Pro | Transformers & Sentence-BERT")
