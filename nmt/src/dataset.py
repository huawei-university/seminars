# Created by: c00k1ez (https://github.com/c00k1ez)


from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset

from .tokenizer import Tokenizer

class NMTDataset(Dataset):

    def __init__(self, 
                 data: List[Tuple[str, str]],
                 source_tokenizer: Tokenizer,
                 target_tokenizer: Tokenizer,
                 source_pad_len: int,
                 target_pad_len: int):
        self.data = data
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_pad_len = source_pad_len
        self.target_pad_len = target_pad_len

    def __len__(self):
        return len(self.data)

    def _check_length_and_pad(self, sample: List[str], pad_len: int) -> Tuple[List[str], List[int]]:
        pad_mask = []
        if len(sample) > pad_len - 2:
            sample = sample[:pad_len - 2]
            sample = ['<BOS>'] + sample + ['<EOS>']
            pad_mask = [1] * len(sample)
        else:
            pad_mask = [1] * (len(sample) + 2) + [0] * (pad_len - 2 - len(sample))
            sample = ['<BOS>'] + sample + ['<EOS>'] + ['<PAD>'] * (pad_len - 2 - len(sample))
        assert len(sample) == pad_len == len(pad_mask)
        return sample, pad_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.LongTensor]:
        source, target = self.data[idx]
        source = self.source_tokenizer.tokenize(source, add_special_tokens=False)
        target = self.target_tokenizer.tokenize(target, add_special_tokens=False)
        source, source_mask = self._check_length_and_pad(source, self.source_pad_len)
        target, target_mask = self._check_length_and_pad(target, self.target_pad_len)
        loss_target = target[1:] + ['<PAD>']
        source = self.source_tokenizer.encode(source)
        target = self.target_tokenizer.encode(target)
        loss_target = self.target_tokenizer.encode(loss_target)

        source = torch.LongTensor(source)
        source_mask = torch.LongTensor(source_mask)
    
        target = torch.LongTensor(target)
        target_mask = torch.LongTensor(target_mask)
        loss_target = torch.LongTensor(loss_target)

        return {
            'source_sentence': source,
            'source_sentence_mask': source_mask,
            'target_language_sentence': target,
            'target_sentence_mask': target_mask,
            'target_for_loss': loss_target
        }
