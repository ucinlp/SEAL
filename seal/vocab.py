"""
Vocab implementation
"""
import pickle
from typing import Dict, List

from spacy.lang import en


_NLP = en.English()
DEFAULT_TOKENIZER = _NLP.tokenizer


def character_tokenizer(string: str):
    return iter(string)


class Vocab:
    """Simple and inflexible vocabulary implementation."""
    def __init__(
        self,
        tokenizer = DEFAULT_TOKENIZER,
    ) -> None:
        self.tokenizer = tokenizer
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}

    def encode(self, string: str) -> List[int]:
        tokens = [token.text for token in self.tokenizer(string)]
        for token in tokens:
            if token not in self.label2id:
                id_ = len(self.label2id)
                self.label2id[token] = id_
                self.id2label[id_] = token
        return [self.label2id[token] for token in tokens]

    def decode(self, ids: List[int]) -> str:
        return ' '.join([self.id2label[id_] for id_ in ids])

    def save(self, path: str) -> None:
        with open(path + '.vocab', 'wb') as f:
            pickle.dump((self.tokenizer, self.label2id, self.id2label), f)

    @classmethod
    def load(cls, path: str) -> 'Vocab':
        vocab = Vocab(tokenizer=character_tokenizer)
        with open(path + '.vocab', 'rb') as f:
            vocab.tokenizer, vocab.label2id, vocab.id2label = pickle.load(f)
        return vocab

