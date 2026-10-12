from canny_chunker import CannySemanticChunker
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def run_tests():
    logger.info("=== Starting Token Count Tests ===")
    
    text = "این یک متن تستی برای شمارش توکن است. بررسی تفاوت بین split و tokenizer."
    
    # 1. Test Approximate Count (Default)
    logger.info("\n--- Test 1: Approximate Count (split) ---")
    chunker_approx = CannySemanticChunker(
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        language='fa',
        exact_token_count=False
    )
    
    # We can access internal method for testing, or just run chunk_with_metadata
    count_approx = chunker_approx._count_tokens(text)
    logger.info(f"Text: '{text}'")
    logger.info(f"Approximate Count (split): {count_approx}")
    
    # 2. Test Exact Count (Tokenizer)
    logger.info("\n--- Test 2: Exact Count (Tokenizer) ---")
    chunker_exact = CannySemanticChunker(
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        language='fa',
        exact_token_count=True
    )
    
    count_exact = chunker_exact._count_tokens(text)
    logger.info(f"Exact Count (Tokenizer): {count_exact}")
    
    # Verify logical difference
    # Split count: "این" "یک" "متن" "تستی" "برای" "شمارش" "توکن" "است." "بررسی" "تفاوت" "بین" "split" "و" "tokenizer." -> 14 words?
    # Tokenizer might be more because of subwords or special tokens.
    
    if count_exact != count_approx:
        logger.info(f"Difference detected: {abs(count_exact - count_approx)} tokens")
    else:
        logger.info("Counts are identical (coincidence or simple text).")

    # 3. Test Chunking with Exact Count
    logger.info("\n--- Test 3: Chunking with Exact Count ---")
    results = chunker_exact.chunk_with_metadata(text)
    for r in results:
        logger.info(f"Chunk Token Count in Metadata: {r.token_count}")
        
    logger.info("Token count tests completed.")

if __name__ == "__main__":
    run_tests()
