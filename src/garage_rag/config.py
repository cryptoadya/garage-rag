"""Reproducible index settings; a directory holds exactly one configuration."""

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version


@dataclass(frozen=True)
class IndexConfig:
    model: str = "intfloat/multilingual-e5-small"
    revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    dimension: int = 384
    chunk_tokens: int = 384
    overlap_tokens: int = 48
    max_sequence_tokens: int = 512
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    normalize: bool = True
    pipeline_version: int = 1

    def __post_init__(self):
        if not 16 <= self.chunk_tokens <= self.max_sequence_tokens <= 512:
            raise ValueError("chunk_tokens must be between 16 and the 512-token model limit")
        if not 0 <= self.overlap_tokens < self.chunk_tokens // 2:
            raise ValueError("overlap_tokens must be non-negative and less than half chunk_tokens")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")

    def as_dict(self):
        return {
            **asdict(self),
            "libraries": {
                name: version(name)
                for name in [
                    "sentence-transformers",
                    "transformers",
                    "pypdf",
                    "fonttools",
                ]
            },
        }

    @property
    def fingerprint(self):
        return digest(self.as_dict())


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
