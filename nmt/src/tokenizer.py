# Created by: c00k1ez (https://github.com/c00k1ez)

from typing import List, Union, Tuple, Dict
from collections import Counter
import math

import nltk

import tqdm



class Tokenizer:

    def __init__(self, language_name: str, vocab_file: str, lower_case: bool = True):
        self.language_name = language_name
        self.vocab_file = vocab_file
        self.lower_case = lower_case
        self.token2id, self.id2token = self._read_vocab()

    def _read_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        token2id = {}
        id2token = []
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    continue 
                line = line.split()
                token2id[line[1]] = int(line[0])
                id2token.append(line[1])
        return token2id, id2token

    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[str]:
        if self.lower_case is True:
            text = text.lower()
        tokenized = nltk.word_tokenize(text)
        checked = []
        for token in tokenized:
            if token not in self.token2id:
                checked.append('<UNK>')  
            else:
                checked.append(token)
        if add_special_tokens is True:
            checked = ['<BOS>'] + checked + ['<EOS>']
        return checked
    
    def encode(self, tokenized: List[str]) -> List[int]:
        encoded = [self.token2id[token] for token in tokenized]
        return encoded
    
    def decode(self, encoded: List[int]) -> List[str]:
        decoded = [self.id2token[idx] for idx in encoded]
        return decoded
    
    def get_vocab_size(self) -> int:
        return len(self.id2token)

    @staticmethod
    def build_vocab(raw_text: List[str], 
                    vocab_file: str, 
                    threshold: float = 1.0,
                    lower_case: bool = True,
                    bos_id: int = 0,
                    eos_id: int = 1,
                    unk_id: int = 2,
                    pad_id: int = 3) -> Union[str, None]:
        vocab = {
            '<BOS>': bos_id, 
            '<EOS>': eos_id,
            '<UNK>': unk_id, 
            '<PAD>': pad_id
        }
        list_of_tokens = []
        for txt in tqdm.tqdm(raw_text):
            if lower_case is True:
                txt = txt.lower()
            tokenized = nltk.word_tokenize(txt)
            list_of_tokens.extend(tokenized)

        print('Get text with {} tokens'.format(len(list_of_tokens)))

        token_cnt = Counter(list_of_tokens)
        vocab_len = int(math.ceil(threshold * len(token_cnt)))
        print('Build vocabulary with {}/{} most common tokens'.format(vocab_len, len(token_cnt)))
        most_common_tokens = token_cnt.most_common(vocab_len)

        vocab.update({token[0]: idx for idx, token in enumerate(most_common_tokens, 4)})

        with open(vocab_file, 'w', encoding='utf-8') as f:
            for token, idx in vocab.items():
                f.write('{}\t{}\n'.format(idx, token))
        print('Write {} file'.format(vocab_file))

        