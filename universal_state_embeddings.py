"""Universal State Embedding System - Fallback without FAISS"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import json

class UniversalStateEmbedding:
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_dir='embedding_cache'):
        """
        Initialize embedding system without FAISS dependency
        Uses numpy for similarity search instead
        """
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Storage - using numpy arrays instead of FAISS
        self.embeddings_matrix = None  # Will be numpy array
        self.state_embeddings = {}  # state_id -> embedding
        self.state_texts = {}  # state_id -> original text
        self.state_outcomes = defaultdict(lambda: defaultdict(list))  # state_id -> action -> outcomes
        
        # Thresholds
        self.similarity_threshold = 0.99  # States above this are "same"
        self.neighbor_threshold = 0.99   # States above this are "similar"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.next_state_id = 0
        self.load_cache()
    
    def get_state_embedding(self, state_text: str) -> np.ndarray:
        """Generate embedding for state"""
        # Normalize text
        state_text = state_text.strip().lower()
        
        # Generate embedding
        embedding = self.encoder.encode(state_text, convert_to_numpy=True)
        
        # L2 normalize for cosine similarity
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def find_similar_states(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Find k most similar states using numpy instead of FAISS
        Returns: [(state_id, similarity_score), ...]
        """
        if self.embeddings_matrix is None or len(self.state_embeddings) == 0:
            return []
        
        # Ensure query is normalized
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings_matrix, query_norm)
        
        # Get top k indices
        k_actual = min(k, len(similarities))
        top_k_indices = np.argpartition(similarities, -k_actual)[-k_actual:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        
        results = []
        for idx in top_k_indices:
            if similarities[idx] > 0:  # Filter out zero similarities
                results.append((int(idx), float(similarities[idx])))
        
        return results
    
    def add_state(self, state_text: str, action: str, outcome: Dict) -> int:
        """
        Add new state-action-outcome to memory
        Returns: state_id (new or existing)
        """
        embedding = self.get_state_embedding(state_text)
        
        # Check if similar state exists
        similar_states = self.find_similar_states(embedding, k=1)
        
        if similar_states and similar_states[0][1] > self.similarity_threshold:
            # Use existing state
            state_id = similar_states[0][0]
        else:
            # Create new state
            state_id = self.next_state_id
            self.next_state_id += 1
            
            # Store embedding
            self.state_embeddings[state_id] = embedding
            self.state_texts[state_id] = state_text
            
            # Update embeddings matrix
            self._update_embeddings_matrix()
        
     
        # Store outcome WITH task embedding
        if 'task' in outcome:
            outcome['task_embedding'] = self.encoder.encode(outcome['task'], convert_to_numpy=True)
        self.state_outcomes[state_id][action].append(outcome)
        
        return state_id
    
    def _update_embeddings_matrix(self):
        """Update the numpy matrix of embeddings"""
        if len(self.state_embeddings) == 0:
            self.embeddings_matrix = None
            return
        
        # Create matrix from all embeddings
        embedding_list = []
        for i in range(len(self.state_embeddings)):
            if i in self.state_embeddings:
                embedding_list.append(self.state_embeddings[i])
            else:
                # Placeholder for missing indices
                embedding_list.append(np.zeros(self.embedding_dim))
        
        self.embeddings_matrix = np.array(embedding_list, dtype=np.float32)
    
    def get_aggregated_knowledge(self, state_text: str, available_actions: List[str]) -> Dict:
        """
        Get aggregated knowledge from similar states
        """
        embedding = self.get_state_embedding(state_text)
        similar_states = self.find_similar_states(embedding, k=20)
        
        knowledge = {
            'exact_match': None,
            'similar_states': [],
            'action_statistics': defaultdict(lambda: {
                'attempts': 0,
                'avg_score': 0.0,
                'success_rate': 0.0,
                'should_avoid': False
            })
        }
        
       
        # Get current task embedding if provided
        current_task_embedding = None
        if task:
            current_task_embedding = self.encoder.encode(task, convert_to_numpy=True)

        # Process similar states
        for state_id, similarity in similar_states:
            # Apply task similarity filtering if we have task embeddings
            if current_task_embedding is not None:
                task_compatible = False
                for action_outcomes in self.state_outcomes.get(state_id, {}).values():
                    for outcome in action_outcomes:
                        if 'task_embedding' in outcome:
                            # Cosine similarity between task embeddings
                            task_sim = np.dot(current_task_embedding, outcome['task_embedding'])
                            if task_sim > 0.7:  # Similar tasks
                                task_compatible = True
                                break
                if not task_compatible:
                    continue  # Skip - different task type
            if similarity > self.similarity_threshold:
                knowledge['exact_match'] = state_id
            
            if similarity > self.neighbor_threshold:
                knowledge['similar_states'].append({
                    'id': state_id,
                    'similarity': similarity,
                    'text': self.state_texts.get(state_id, '')
                })
                
                # Aggregate action outcomes with similarity weighting
                for action, outcomes in self.state_outcomes[state_id].items():
                    if action in available_actions:
                        stats = knowledge['action_statistics'][action]
                        
                        # Weight by similarity
                        weight = similarity
                        
                        for outcome in outcomes:
                            stats['attempts'] += 1
                            score = outcome.get('effectiveness', 0.0)
                            stats['avg_score'] += score * weight
                            
                            if score > 0.5:
                                stats['success_rate'] += weight
        
        # Normalize statistics
        for action, stats in knowledge['action_statistics'].items():
            if stats['attempts'] > 0:
                stats['avg_score'] /= stats['attempts']
                stats['success_rate'] /= stats['attempts']
                
                # Determine if should avoid
                if stats['attempts'] >= 3 and stats['avg_score'] < 0.1:
                    stats['should_avoid'] = True
                elif stats['attempts'] >= 5:
                    stats['should_avoid'] = True
        
        return knowledge
    
    def save_cache(self):
        """Save embeddings and outcomes to disk"""
        cache_file = self.cache_dir / 'embedding_cache.pkl'
        
        cache_data = {
            'embeddings_matrix': self.embeddings_matrix,
            'state_embeddings': self.state_embeddings,
            'state_texts': self.state_texts,
            'state_outcomes': dict(self.state_outcomes),
            'next_state_id': self.next_state_id
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def load_cache(self):
        """Load embeddings and outcomes from disk"""
        cache_file = self.cache_dir / 'embedding_cache.pkl'
        
        if cache_file.exists():
        
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.embeddings_matrix = cache_data.get('embeddings_matrix')
            self.state_embeddings = cache_data.get('state_embeddings', {})
            self.state_texts = cache_data.get('state_texts', {})
            self.state_outcomes = defaultdict(lambda: defaultdict(list), 
                                                cache_data.get('state_outcomes', {}))
            self.next_state_id = cache_data.get('next_state_id', 0)
            
            print(f"[EMBEDDINGS] Loaded {len(self.state_texts)} states from cache")


# Global instance
state_embeddings = UniversalStateEmbedding()