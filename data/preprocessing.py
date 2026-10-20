from hashlib import blake2b
from typing import Any, Callable

from datasets import Dataset
from tokenizers.normalizers import BertNormalizer


def deduplicate(dataset: Dataset, key_extractor: Callable[[dict[str, Any]], str]) -> Dataset:
    normalizer = BertNormalizer()
    hashes = set()

    def is_unique(example: dict[str, Any]) -> bool:
        key = key_extractor(example)
        key = normalizer.normalize_str(key)

        key_hash = blake2b(key.encode("utf-8"), digest_size=16).digest()
        if key_hash in hashes:
            return False

        hashes.add(key_hash)
        return True

    return dataset.filter(is_unique)
