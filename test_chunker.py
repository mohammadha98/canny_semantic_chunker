from canny_chunker import CannySemanticChunker
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def run_test():
    # Persian sample text with two distinct topics: Space Exploration and Healthy Diet
    text = (
        "ناسا برنامه‌های جدیدی برای سفر به مریخ دارد. "
        "این سازمان قصد دارد تا سال ۲۰۳۰ فضانوردان را به سیاره سرخ بفرستد. "
        "کاوشگرهای جدیدی برای بررسی سطح مریخ طراحی شده‌اند. "
        "سفر به فضا همیشه یکی از آرزوهای بشر بوده است. "
        "تکنولوژی موشکی به سرعت در حال پیشرفت است. "
        "خوردن سبزیجات و میوه‌ها برای سلامتی بسیار مفید است. "
        "رژیم غذایی سالم باید شامل پروتئین و ویتامین‌های کافی باشد. "
        "ورزش روزانه به کاهش استرس و بهبود سلامت قلب کمک می‌کند. "
        "نوشیدن آب کافی در طول روز برای بدن ضروری است. "
        "خواب کافی نیز یکی از ارکان اصلی زندگی سالم است."
    )

    logger.info("--- Starting Canny Semantic Chunker Test ---")
    
    try:
        # Initialize chunker for Persian (fa)
        chunker = CannySemanticChunker(model_name='paraphrase-multilingual-MiniLM-L12-v2', language='fa')
        
        logger.info("Chunking text...")
        chunks = chunker.chunk(text)
        
        logger.info(f"Total chunks generated: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            logger.info(f"\nChunk {i+1}:")
            logger.info(f"'{chunk}'")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run_test()
