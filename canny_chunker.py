import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pysbd
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ChunkResult:
    text: str
    sentences: List[str]
    sentence_indices: Tuple[int, int]
    token_count: int
    centroid: np.ndarray
    avg_internal_distance: float
    split_reason: str

class CannySemanticChunker:
    """
    Semantic Chunker using Bi-Encoder for initial segmentation and optional Cross-Encoder for verification.
    
    Attributes:
        use_cross_encoder (bool): Enable hybrid mode with Cross-Encoder for gray zone verification.
        cross_encoder_model (str): Model name for Cross-Encoder.
        cross_encoder_threshold (float): Threshold for Cross-Encoder similarity (below this = split).
        cross_encoder_context_sentences (int): Number of context sentences to use for Cross-Encoder.
    """
    def __init__(
        self, 
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', 
        language: str = 'en', 
        adaptive_threshold: bool = True,
        low_percentile: int = 60,
        high_percentile: int = 85,
        manual_low_threshold: Optional[float] = None,
        manual_high_threshold: Optional[float] = None,
        min_sentences: int = 2,
        max_sentences: int = 15,
        min_tokens: int = 50,
        max_tokens: int = 512,
        centroid_window_size: int = 5,
        exact_token_count: bool = False,
        # Cross-Encoder settings
        use_cross_encoder: bool = False,
        cross_encoder_model: str = "cross-encoder/ms-marco-multilingual-MiniLM-L6-v2",
        cross_encoder_threshold: float = 0.5,
        cross_encoder_context_sentences: int = 3
    ):
        self.model_name = model_name
        self._model = None  # Lazy loading
        self._cross_encoder = None # Lazy loading for Cross-Encoder
        self.seg = pysbd.Segmenter(language=language, clean=False)
        self.adaptive_threshold = adaptive_threshold
        
        # Threshold settings
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.manual_low_threshold = manual_low_threshold
        self.manual_high_threshold = manual_high_threshold
        
        # Fallback default thresholds (if adaptive is off and manual is not set)
        self.default_low_threshold = 0.6
        self.default_high_threshold = 0.8
        
        # Size constraints
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        
        # Centroid settings
        self.centroid_window_size = centroid_window_size

        # Token count settings
        self.exact_token_count = exact_token_count

        # Cross-Encoder settings
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_model_name = cross_encoder_model
        self.cross_encoder_threshold = cross_encoder_threshold
        self.cross_encoder_context_sentences = cross_encoder_context_sentences

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def cross_encoder(self):
        if self._cross_encoder is None and self.use_cross_encoder:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self.cross_encoder_model_name)
        return self._cross_encoder

    def _count_tokens(self, text: str) -> int:
        if self.exact_token_count:
            # Using model's tokenizer for exact count (includes special tokens)
            return len(self.model.tokenizer.encode(text))
        return len(text.split())

    def _calculate_cosine_distance(self, embedding1, embedding2):
        sim = cosine_similarity([embedding1], [embedding2])[0][0]
        return 1 - sim

    def _get_thresholds(self, distances: List[float]) -> Tuple[float, float]:
        if self.manual_low_threshold is not None and self.manual_high_threshold is not None:
            return self.manual_low_threshold, self.manual_high_threshold
            
        if not distances:
            return self.default_low_threshold, self.default_high_threshold
            
        if self.adaptive_threshold:
            low_threshold = np.percentile(distances, self.low_percentile)
            high_threshold = np.percentile(distances, self.high_percentile)
            return low_threshold, high_threshold
            
        return self.default_low_threshold, self.default_high_threshold
        
    def get_thresholds(self, text: str) -> Tuple[float, float]:
        """For debugging purposes"""
        sentences = self.seg.segment(text)
        if len(sentences) < 2:
            return self.default_low_threshold, self.default_high_threshold
            
        embeddings = self.model.encode(sentences, batch_size=32, show_progress_bar=False)
        distances = []
        for i in range(len(embeddings) - 1):
            dist = self._calculate_cosine_distance(embeddings[i], embeddings[i+1])
            distances.append(dist)
            
        return self._get_thresholds(distances)

    def chunk(self, text: str) -> List[str]:
        results = self.chunk_with_metadata(text)
        return [r.text for r in results]

    def chunk_with_metadata(self, text: str) -> List[ChunkResult]:
        # 1. Segment sentences
        sentences = self.seg.segment(text)
        if not sentences:
            return []
        
        # Pre-calculate token counts for all sentences
        # This optimizes performance, especially if exact_token_count is True
        sentence_token_counts = [self._count_tokens(s) for s in sentences]
        
        # Handle single sentence case
        if len(sentences) == 1:
            embeddings = self.model.encode(sentences, batch_size=32, show_progress_bar=False)
            return [ChunkResult(
                text=sentences[0],
                sentences=sentences,
                sentence_indices=(0, 0),
                token_count=sentence_token_counts[0],
                centroid=embeddings[0],
                avg_internal_distance=0.0,
                split_reason="end_of_text"
            )]

        # 2. Compute embeddings
        embeddings = self.model.encode(sentences, batch_size=32, show_progress_bar=False)

        # 3. Calculate consecutive distances
        distances = []
        for i in range(len(embeddings) - 1):
            dist = self._calculate_cosine_distance(embeddings[i], embeddings[i+1])
            distances.append(dist)

        # 4. Determine Thresholds
        low_threshold, high_threshold = self._get_thresholds(distances)

        chunks: List[ChunkResult] = []
        
        # State for current chunk
        current_chunk_sentences = [sentences[0]]
        current_chunk_embeddings = [embeddings[0]]
        current_start_idx = 0
        current_tokens = sentence_token_counts[0] # Initialize accumulator
        
        # Loop through sentences
        for i in range(len(distances)):
            dist = distances[i]
            next_sentence = sentences[i+1]
            next_embedding = embeddings[i+1]
            next_tokens = sentence_token_counts[i+1]
            
            # Current chunk stats
            current_sentences_count = len(current_chunk_sentences)
            
            # Logic for splitting
            should_split = False
            split_reason = ""
            
            # 1. Max constraints
            # Check if ADDING next sentence would exceed limits
            if current_sentences_count >= self.max_sentences:
                should_split = True
                split_reason = "max_size"
            elif current_tokens + next_tokens > self.max_tokens:
                should_split = True
                split_reason = "max_size"
            else:
                # 2. Threshold Logic
                is_min_satisfied = (current_sentences_count >= self.min_sentences) and \
                                   (current_tokens >= self.min_tokens)
                
                threshold_split = False
                threshold_reason = ""
                
                if dist < low_threshold:
                    threshold_split = False
                elif dist > high_threshold:
                    threshold_split = True
                    threshold_reason = "threshold"
                else:
                    # Gray Zone Logic
                    if self.use_cross_encoder:
                        # Cross-Encoder Verification
                        # Build context from last N sentences of current chunk
                        context_sentences = current_chunk_sentences[-self.cross_encoder_context_sentences:]
                        context = " ".join(context_sentences)
                        
                        # Get similarity score from Cross-Encoder
                        # predict expects list of pairs: [(context, next_sentence)]
                        score = self.cross_encoder.predict([(context, next_sentence)])[0]
                        
                        # In Cross-Encoder, lower score means less similar -> Split
                        if score < self.cross_encoder_threshold:
                            threshold_split = True
                            threshold_reason = "cross_encoder_boundary"
                        else:
                            threshold_split = False
                    else:
                        # Existing Centroid-based Drift Detection
                        if len(current_chunk_embeddings) > self.centroid_window_size:
                            window_embeddings = current_chunk_embeddings[-self.centroid_window_size:]
                            centroid = np.mean(window_embeddings, axis=0)
                        else:
                            centroid = np.mean(current_chunk_embeddings, axis=0)
                            
                        dist_to_centroid = self._calculate_cosine_distance(next_embedding, centroid)
                        
                        if dist_to_centroid > high_threshold:
                            threshold_split = True
                            threshold_reason = "semantic_drift"
                        else:
                            threshold_split = False
                
                if threshold_split:
                    if is_min_satisfied:
                        should_split = True
                        split_reason = threshold_reason
                    else:
                        should_split = False
                else:
                    should_split = False

            if should_split:
                full_text = " ".join(current_chunk_sentences)
                final_centroid = np.mean(current_chunk_embeddings, axis=0)
                
                if len(current_chunk_embeddings) > 1:
                    dists = [self._calculate_cosine_distance(emb, final_centroid) for emb in current_chunk_embeddings]
                    avg_dist = float(np.mean(dists))
                else:
                    avg_dist = 0.0
                
                chunk_res = ChunkResult(
                    text=full_text,
                    sentences=current_chunk_sentences.copy(), # Good practice
                    sentence_indices=(current_start_idx, i),
                    token_count=current_tokens, # Use accumulator
                    centroid=final_centroid,
                    avg_internal_distance=avg_dist,
                    split_reason=split_reason
                )
                chunks.append(chunk_res)
                
                # Start new chunk
                current_chunk_sentences = [next_sentence]
                current_chunk_embeddings = [next_embedding]
                current_tokens = next_tokens # Reset accumulator
                current_start_idx = i + 1
            else:
                current_chunk_sentences.append(next_sentence)
                current_chunk_embeddings.append(next_embedding)
                current_tokens += next_tokens # Update accumulator

        # Add the last chunk
        if current_chunk_sentences:
            full_text = " ".join(current_chunk_sentences)
            final_centroid = np.mean(current_chunk_embeddings, axis=0)
            if len(current_chunk_embeddings) > 1:
                dists = [self._calculate_cosine_distance(emb, final_centroid) for emb in current_chunk_embeddings]
                avg_dist = float(np.mean(dists))
            else:
                avg_dist = 0.0
                
            chunk_res = ChunkResult(
                text=full_text,
                sentences=current_chunk_sentences,
                sentence_indices=(current_start_idx, len(sentences)-1),
                token_count=current_tokens, # Use accumulator
                centroid=final_centroid,
                avg_internal_distance=avg_dist,
                split_reason="end_of_text"
            )
            chunks.append(chunk_res)

        return chunks
