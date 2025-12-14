"""
MINES PATTERNS FROM TRAINING DATA - EVOLVES WITH USE
"""
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import faiss

class AdvancedPatternMiner:
    def __init__(self):
        self.pattern_db = "evolution/patterns.faiss"
        self.metadata_db = "evolution/patterns_metadata.pkl"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.pattern_metadata = []
        self.pattern_strength = defaultdict(float)
        
        self.load_patterns()
    
    def load_patterns(self):
        """Load existing patterns"""
        if Path(self.pattern_db).exists():
            self.index = faiss.read_index(self.pattern_db)
            with open(self.metadata_db, 'rb') as f:
                self.pattern_metadata = pickle.load(f)
    
    def mine_patterns(self, data: Any, context: Dict = None) -> List[Dict]:
        """Mine patterns from any data type"""
        # Convert data to text representation
        text_representation = self._to_text(data)
        
        # Generate embedding
        embedding = self.embedding_model.encode([text_representation])[0]
        
        # Search for similar patterns
        similar_patterns = self._find_similar_patterns(embedding)
        
        if similar_patterns:
            # Reinforce existing patterns
            for pattern_id, similarity in similar_patterns:
                self._reinforce_pattern(pattern_id, similarity)
            
            return [self.pattern_metadata[pid] for pid, _ in similar_patterns]
        else:
            # New pattern discovered
            new_pattern = self._create_new_pattern(embedding, text_representation, context)
            return [new_pattern]
    
    def _find_similar_patterns(self, embedding: np.ndarray, threshold: float = 0.8):
        """Find similar patterns using FAISS vector search"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Search index
        embedding = embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(embedding, 5)
        
        similar = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and 1 - dist > threshold:  # Convert distance to similarity
                similar.append((idx, 1 - dist))
        
        return similar
    
    def _create_new_pattern(self, embedding: np.ndarray, text: str, context: Dict) -> Dict:
        """Create and store new pattern"""
        pattern_id = len(self.pattern_metadata)
        
        new_pattern = {
            'id': pattern_id,
            'embedding': embedding,
            'text': text[:1000],  # Store truncated text
            'context': context,
            'strength': 1.0,
            'discovered_at': datetime.now().isoformat(),
            'usage_count': 0,
            'success_rate': 0.0
        }
        
        # Add to index
        if self.index is None:
            dimension = embedding.shape[0]
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        self.pattern_metadata.append(new_pattern)
        
        # Save
        self._save_patterns()
        
        return new_pattern
    
    def evolve_mining(self, success_metrics: Dict):
        """Evolve pattern mining based on success"""
        # Adjust similarity thresholds
        # Update embedding model if needed
        # Prune weak patterns
        self._prune_weak_patterns(threshold=0.3)
        
        # Increase pattern recognition complexity
        self._increase_complexity()
