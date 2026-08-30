try:
    import faiss as _faiss_module
    FAISS_AVAILABLE = True
except ImportError:
    _faiss_module = None
    FAISS_AVAILABLE = False
    print("[WARNING] faiss not found. Semantic search will be disabled. Gallery, Chat and all other features will still work.")

import numpy as np
import os
import pickle
from database import get_event_by_id

class VectorStore:
    """
    Manages the FAISS index for semantic video frame retrieval.
    Strictly maps faiss_vector_id -> event_id.
    Falls back gracefully if faiss is not installed.
    """
    
    def __init__(self, index_path="crimevision_faiss.index", meta_path="crimevision_faiss_meta.pkl", dim=512):

        self.index_path = index_path
        self.meta_path = meta_path
        self.dim = dim
        self.index = None
        self.metadata = []
        
        if not FAISS_AVAILABLE:
            print("[VectorStore] FAISS unavailable — running in stub mode. Semantic search disabled.")
            return
        
        faiss = _faiss_module
        
        # Load or create FAISS index
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f) # List of event_ids
            print(f"Loaded existing FAISS index with {self.index.ntotal} vectors.")
        else:
            # Using Inner Product since CLIP embeddings are L2 normalized (Cosine Similarity)
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = [] # List mapping faiss index -> event_id
            print("Created new FAISS index.")

    def add_embedding(self, embedding, event_id):
        """
        Adds a single embedding and its associated event_id to the store.
        """
        if not FAISS_AVAILABLE or self.index is None or embedding is None:
            return
            
        # Add vector to FAISS
        # FAISS expects a 2D numpy array (1, dim)
        vector = np.array([embedding]).astype('float32')
        self.index.add(vector)
        
        # Add metadata mapping (just the event_id)
        self.metadata.append(event_id)

    def search(self, query_embedding, k=10):
        """
        Searches the FAISS index for the top-k nearest embeddings.
        Returns a list of full event records from the database with confidence scores.
        """
        if not FAISS_AVAILABLE or self.index is None:
            return []
        if query_embedding is None or self.index.ntotal == 0:
            return []
            
        vector = np.array([query_embedding]).astype('float32')
        
        # Perform search
        D, I = self.index.search(vector, k)
        
        results = []
        for i in range(len(I[0])):
            idx = I[0][i]
            if idx != -1 and idx < len(self.metadata): # Valid match
                # Scale cosine similarity (roughly -1 to 1) to a percentage
                similarity = float(D[0][i])
                
                # CLIP cosine similarities are usually between 0.20 and 0.35 for good matches
                # We can enforce a minimum threshold to drop completely irrelevant matches
                if similarity < 0.24:
                    continue
                    
                confidence = max(0, min(1, similarity))
                
                event_id = self.metadata[idx]
                event_record = get_event_by_id(event_id)
                
                if event_record:
                    event_record["similarity"] = confidence
                    results.append(event_record)
                
        return results

    def save(self):
        """
        Persists the index and metadata to disk.
        """
        if not FAISS_AVAILABLE or self.index is None:
            return
        faiss = _faiss_module
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def delete_by_event_ids(self, event_ids_to_remove):
        """
        Removes vectors associated with a set of event_ids by rebuilding the index.
        """
        if not FAISS_AVAILABLE or self.index is None:
            return
        if not event_ids_to_remove or self.index.ntotal == 0:
            return
            
        faiss = _faiss_module
        remove_set = set(event_ids_to_remove)
        
        new_index = faiss.IndexFlatIP(self.dim)
        new_metadata = []
        
        for i in range(self.index.ntotal):
            evt_id = self.metadata[i]
            if evt_id not in remove_set:
                # faiss.IndexFlat supports reconstruct(i) to get the original vector
                vec = self.index.reconstruct(i)
                new_index.add(np.array([vec]).astype('float32'))
                new_metadata.append(evt_id)
                
        self.index = new_index
        self.metadata = new_metadata
        self.save()
