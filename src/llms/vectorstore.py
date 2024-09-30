import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSEmbeddingsSearch:

    def __init__(self, use_gpu: bool = False):

        self.device = 'cuda' if use_gpu is True else 'cpu'
        self.model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.embeddings_model = SentenceTransformer(self.model_name)
        self.tokenizer = None
        self.index = None
        self.d = None

    def get_embeddings(self, texts):

        embeddings = self.embeddings_model.encode([texts], convert_to_tensor=False, device=self.device)
        self.d = embeddings.shape[1]
        return np.array(embeddings)

    def create_faiss_index(self, embeddings):
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(self.d)
        index.add(embeddings)

        self.index = index

    def get_query_result(self, query_word: str, k: int, query_string: str=''):
        
        query_emb = self.get_embeddings(query_word)
        if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(query_emb)
        
        dist, ind = self.index.search(query_emb, k)
        return dist, ind
    
    def save_index(self, path: str):
        faiss.write_index(self.index, path)
