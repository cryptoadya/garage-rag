"""Pinned, isolated embedding profiles; baseline configuration stays byte-for-byte stable."""

from dataclasses import replace

from .config import IndexConfig

BASE_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


def profile_config(name, **kwargs):
    config = IndexConfig(**kwargs)
    if name == "baseline":
        return config
    if name == "e5-base":
        return replace(
            config, model="intfloat/multilingual-e5-base", revision=BASE_REVISION, dimension=768
        )
    if name == "bge-m3":
        return replace(
            config,
            model="BAAI/bge-m3",
            revision=BGE_REVISION,
            dimension=1024,
            query_prefix="",
            passage_prefix="",
        )
    raise ValueError("Unknown embedding profile")
