import logging
from canny_chunker import CannySemanticChunker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def print_chunks(chunks, title):
    logger.info(f"\n=== {title} ===")
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1}:")
        logger.info(f"  Reason: {chunk.split_reason}")
        logger.info(f"  Sentences: {len(chunk.sentences)}")
        logger.info(f"  Text: {chunk.text[:100]}...")
        logger.info("-" * 40)

def run_tests():
    # A text with topic shifts about Artificial Intelligence and Python
    text = """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans.
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving".
    This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
    
    Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.
    Recently, artificial neural networks have been able to surpass many previous approaches in performance.
    Deep learning is a subset of machine learning that uses multi-layered neural networks to solve complex problems.
    
    Python is a high-level, general-purpose programming language.
    Its design philosophy emphasizes code readability with the use of significant indentation.
    Python is dynamically-typed and garbage-collected.
    It supports multiple programming paradigms, including structured, object-oriented and functional programming.
    """
    
    # 1. Test Default Mode (Centroid)
    logger.info("Running Default Mode (Centroid)...")
    chunker_default = CannySemanticChunker(
        model_name='all-MiniLM-L6-v2', # Fast model for test
        use_cross_encoder=False
    )
    chunks_default = chunker_default.chunk_with_metadata(text)
    print_chunks(chunks_default, "Default Mode Results")
    
    # 2. Test Hybrid Mode (Cross-Encoder)
    logger.info("Running Hybrid Mode (Cross-Encoder)...")
    chunker_hybrid = CannySemanticChunker(
        model_name='all-MiniLM-L6-v2',
        use_cross_encoder=True,
        # Using a small English model for speed in testing
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        cross_encoder_threshold=0.5,
        # Force Gray Zone to test Cross-Encoder logic
        adaptive_threshold=False,
        manual_low_threshold=0.01,
        manual_high_threshold=0.99
    )
    chunks_hybrid = chunker_hybrid.chunk_with_metadata(text)
    print_chunks(chunks_hybrid, "Hybrid Mode Results")
    
    # 3. Verification
    logger.info("\n=== Verification ===")
    hybrid_reasons = [c.split_reason for c in chunks_hybrid]
    if "cross_encoder_boundary" in hybrid_reasons:
        logger.info("SUCCESS: 'cross_encoder_boundary' found in split reasons.")
    else:
        logger.info("NOTE: 'cross_encoder_boundary' not triggered. Text might be too cohesive or threshold too low.")
        
    logger.info(f"Default chunks: {len(chunks_default)}")
    logger.info(f"Hybrid chunks: {len(chunks_hybrid)}")

if __name__ == "__main__":
    run_tests()
