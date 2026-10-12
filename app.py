import streamlit as st
import logging
from canny_chunker import CannySemanticChunker

# Page Configuration
st.set_page_config(
    page_title="ADT Semantic Chunker",
    page_icon="🔪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .chunk-box {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .chunk-header {
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    /* Enhance metrics styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.title("🔪 ADT Semantic Chunker")
st.markdown("### (Canny-Inspired Algorithm)")

# --- Sidebar: Settings ---
st.sidebar.header("⚙️ تنظیمات / Settings")

# 1. Model Selection
embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    ["paraphrase-multilingual-MiniLM-L12-v2", "all-MiniLM-L6-v2"],
    index=0
)

# 2. Threshold Mode
threshold_mode = st.sidebar.radio(
    "Threshold Mode",
    ["Adaptive (Percentile)", "Manual"]
)

manual_low_threshold = None
manual_high_threshold = None
low_percentile = 60
high_percentile = 85

if threshold_mode == "Manual":
    tau_low = st.sidebar.slider("τ_low", 0.0, 1.0, 0.3, 0.05)
    tau_high = st.sidebar.slider("τ_high", 0.0, 1.0, 0.6, 0.05)
    manual_low_threshold = tau_low
    manual_high_threshold = tau_high
    adaptive_threshold = False
else:
    low_percentile = st.sidebar.slider("Low Percentile", 0, 100, 60)
    high_percentile = st.sidebar.slider("High Percentile", 0, 100, 85)
    adaptive_threshold = True

# 3. Token Limits
max_tokens = st.sidebar.number_input("Max Tokens per Chunk", 100, 2000, 512)
min_tokens = st.sidebar.number_input("Min Tokens per Chunk", 10, 500, 50)

# 4. Hybrid Mode
use_cross_encoder = st.sidebar.checkbox("🔬 Use Cross-Encoder (Hybrid Mode)", False)
cross_threshold = 0.5
if use_cross_encoder:
    cross_threshold = st.sidebar.slider("Cross-Encoder Threshold", 0.0, 1.0, 0.5)

# Sample Text Data
sample_persian = """هوش مصنوعی یکی از شاخه‌های علوم کامپیوتر است که به ساخت ماشین‌های هوشمند می‌پردازد.
این ماشین‌ها قادر به انجام کارهایی هستند که معمولاً نیاز به هوش انسانی دارند.
یادگیری ماشین زیرشاخه‌ای از هوش مصنوعی است. در این روش، سیستم‌ها از داده‌ها یاد می‌گیرند.
الوریتم‌های یادگیری عمیق از شبکه‌های عصبی چندلایه استفاده می‌کنند.
پردازش زبان طبیعی حوزه دیگری است که به تعامل بین کامپیوتر و زبان انسان می‌پردازد.
این فناوری در ترجمه ماشینی و چت‌بات‌ها کاربرد دارد.
"""

sample_english = """Artificial intelligence is transforming how we interact with technology.
Machine learning algorithms can now recognize patterns in vast amounts of data.
Natural language processing has made significant advances in recent years.
Large language models can generate human-like text and answer complex questions.
Computer vision enables machines to interpret and understand visual information.
These technologies are being applied in healthcare, autonomous vehicles, and more.
"""

# Sample Text Buttons
if st.sidebar.button("📄 Load Sample Text (Persian)"):
    st.session_state['input_text'] = sample_persian
if st.sidebar.button("📄 Load Sample Text (English)"):
    st.session_state['input_text'] = sample_english

# --- Main Area ---
col1, col2 = st.columns([1, 1])

# Left Column: Input
with col1:
    st.subheader("📝 متن ورودی / Input Text")
    input_text_val = st.session_state.get('input_text', "")
    input_text = st.text_area(
        "متن خود را وارد کنید:",
        value=input_text_val,
        height=400,
        placeholder="متن فارسی یا انگلیسی خود را اینجا وارد کنید..."
    )
    process_button = st.button("🚀 پردازش / Process", type="primary", use_container_width=True)

# Caching the Chunker
@st.cache_resource
def load_chunker(
    model_name, 
    adaptive_threshold,
    low_percentile,
    high_percentile,
    manual_low_threshold,
    manual_high_threshold,
    min_tokens,
    max_tokens,
    use_cross_encoder,
    cross_threshold
):
    """Cache the chunker to avoid reloading model on every interaction"""
    return CannySemanticChunker(
        model_name=model_name,
        adaptive_threshold=adaptive_threshold,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        manual_low_threshold=manual_low_threshold,
        manual_high_threshold=manual_high_threshold,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        use_cross_encoder=use_cross_encoder,
        cross_encoder_threshold=cross_threshold,
        # Default or reasonable values for others
        cross_encoder_context_sentences=3,
        centroid_window_size=5,
        exact_token_count=False # Keeping default for speed in UI
    )

# Right Column: Output
with col2:
    st.subheader("📦 چانک‌ها / Chunks")
    
    chunks = []
    if process_button and input_text:
        try:
            with st.spinner("در حال پردازش... / Processing..."):
                # Load/Get cached chunker
                chunker = load_chunker(
                    model_name=embedding_model,
                    adaptive_threshold=adaptive_threshold,
                    low_percentile=low_percentile,
                    high_percentile=high_percentile,
                    manual_low_threshold=manual_low_threshold,
                    manual_high_threshold=manual_high_threshold,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    use_cross_encoder=use_cross_encoder,
                    cross_threshold=cross_threshold
                )
                
                chunks = chunker.chunk_with_metadata(input_text)
                
                if not chunks:
                    st.warning("هیچ چانکی تولید نشد. متن ورودی ممکن است خیلی کوتاه باشد.")
                
                for i, chunk in enumerate(chunks):
                    # Using expander for cleaner look
                    expander_label = f"Chunk {i+1} | {chunk.token_count} tokens | {chunk.split_reason}"
                    with st.expander(expander_label, expanded=True):
                        st.markdown(f"**Text:**\n\n{chunk.text}")
                        st.caption(f"Sentences: {len(chunk.sentences)} | Split Reason: {chunk.split_reason}")
                        
        except Exception as e:
            st.error(f"❌ خطا در پردازش: {str(e)}")
            st.info("لطفاً متن ورودی را بررسی کنید یا تنظیمات را تغییر دهید.")

# --- Statistics Bar (Bottom) ---
if chunks:
    st.divider()
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        st.metric("📦 Total Chunks", len(chunks))
    
    with stat_cols[1]:
        avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
        st.metric("📏 Avg Tokens", f"{avg_tokens:.0f}")
    
    with stat_cols[2]:
        total_sentences = sum(len(c.sentences) for c in chunks)
        st.metric("📝 Total Sentences", total_sentences)
    
    with stat_cols[3]:
        st.metric("⚡ Mode", "Hybrid" if use_cross_encoder else "Standard")
