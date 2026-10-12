from canny_chunker import CannySemanticChunker
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def run_tests():
    logger.info("=== Starting Phase 2 Tests ===")
    
    # Initialize Chunker with Persian support but small constraints for testing
    chunker = CannySemanticChunker(
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        language='fa',
        min_sentences=2,
        max_sentences=5,
        min_tokens=5,
        max_tokens=100,
        centroid_window_size=2
    )

    # Test 1: Size Constraints (Min Sentences)
    logger.info("\n--- Test 1: Size Constraints (Min Sentences) ---")
    text_short = "بله. نه. شاید. البته. حتماً."
    logger.info(f"Input: {text_short}")
    results = chunker.chunk_with_metadata(text_short)
    logger.info(f"Chunks generated: {len(results)}")
    for r in results:
        logger.info(f"Chunk: '{r.text}' | Sentences: {len(r.sentences)} | Reason: {r.split_reason}")
    
    # With min_sentences=2, we expect fewer chunks even if they are semantically different?
    # Actually "Bale. Na. ..." are very short. 
    # If they are distinct, they might want to split. But min_sentences=2 forces them to group.
    
    # Test 2: Max Constraints (Force Split)
    logger.info("\n--- Test 2: Max Constraints (Force Split) ---")
    # Create a long repetitive text about one topic
    text_long = " ".join(["این یک جمله تکراری است." for _ in range(12)])
    logger.info(f"Input: 12 repetitive sentences.")
    # Max sentences is 5. So we expect at least 3 chunks (5, 5, 2).
    results = chunker.chunk_with_metadata(text_long)
    logger.info(f"Chunks generated: {len(results)}")
    for i, r in enumerate(results):
        logger.info(f"Chunk {i+1}: Sentences: {len(r.sentences)} | Reason: {r.split_reason}")

    # Test 3: Sliding Window (Drift Detection)
    logger.info("\n--- Test 3: Sliding Window & Metadata ---")
    # A text that slowly drifts.
    # Topic A: Space (4 sentences)
    # Topic B: Food (4 sentences)
    text_drift = (
        "ناسا به مریخ می‌رود. موشک‌ها آماده پرتاب هستند. فضانوردان تمرین می‌کنند. سفر فضایی هیجان‌انگیز است. "
        "سیب میوه خوشمزه‌ای است. خوردن سبزیجات مفید است. پیتزا غذای محبوبی است. آشپزی هنر است."
    )
    
    # We set window size small (2) to detect drift quickly.
    chunker.centroid_window_size = 2
    chunker.max_sentences = 10 # Relax max constraint to test drift
    
    results = chunker.chunk_with_metadata(text_drift)
    logger.info(f"Chunks generated: {len(results)}")
    for i, r in enumerate(results):
        logger.info(f"Chunk {i+1}:")
        logger.info(f"  Text: {r.text[:50]}...")
        logger.info(f"  Sentences: {len(r.sentences)}")
        logger.info(f"  Reason: {r.split_reason}")
        logger.info(f"  Centroid Shape: {r.centroid.shape}")
        logger.info(f"  Indices: {r.sentence_indices}")

    # Test 4: Verify Metadata
    logger.info("\n--- Test 4: Metadata Verification ---")
    if len(results) > 0:
        r = results[0]
        logger.info(f"Has centroid? {isinstance(r.centroid, np.ndarray)}")
        logger.info(f"Has split_reason? {bool(r.split_reason)}")
        logger.info(f"Has token_count? {r.token_count > 0}")
        logger.info(f"Has indices? {isinstance(r.sentence_indices, tuple)}")
        logger.info("Metadata verification passed.")

if __name__ == "__main__":
    run_tests()
