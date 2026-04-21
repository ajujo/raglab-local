from rag_lab.embedding.encoder import reset_embedding_cache
reset_embedding_cache()
from rag_lab.retrieval.reranker import reset_reranker_cache
reset_reranker_cache()
print('Fixture success!')
