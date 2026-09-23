import functools

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models as qm

from backend import config


class DenseEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=config.FASTEMBED_CACHE_DIR)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()


class SparseEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SparseTextEmbedding(model_name=model_name, cache_dir=config.FASTEMBED_CACHE_DIR)

    @staticmethod
    def _to_vector(emb) -> qm.SparseVector:
        return qm.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

    def embed_documents(self, texts: list[str]) -> list[qm.SparseVector]:
        return [self._to_vector(e) for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> qm.SparseVector:
        return self._to_vector(next(iter(self._model.query_embed(text))))


@functools.lru_cache(maxsize=1)
def get_dense() -> DenseEmbedder:
    return DenseEmbedder(config.EMBED_MODEL)


@functools.lru_cache(maxsize=1)
def get_sparse() -> SparseEmbedder:
    return SparseEmbedder(config.SPARSE_MODEL)
