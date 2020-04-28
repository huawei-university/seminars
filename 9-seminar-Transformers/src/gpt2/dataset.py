# Created by: c00k1ez (https://github.com/c00k1ez)

import torch
from typing import List, Dict
from transformers import GPT2Tokenizer


class DialogueDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data: List[Dict[str, str]], 
                 tokenizer: GPT2Tokenizer, 
                 pad_len: int,
                 ):
        self.data = data
        self.tokenizer = tokenizer
        self.pad_len = pad_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):
        sample = self.data[indx]
        context, answer = sample['context'], sample['answer']

        context = self.tokenizer.encode(context)
        answer = self.tokenizer.encode(answer)

        cntx_token_id, answer_token_id = self.tokenizer.additional_special_tokens_ids
        sample = [self.tokenizer.bos_token_id] + \
                 [cntx_token_id] + context + \
                 [answer_token_id] + answer + \
                 [self.tokenizer.eos_token_id]
        assert len(sample) <= self.pad_len
        mask = [1] * len(sample) + [0] * (self.pad_len - len(sample))
        label = sample + [-100] * (self.pad_len - len(sample))
        sample = sample + [self.tokenizer.bos_token_id] * (self.pad_len - len(sample))

        sample = torch.LongTensor(sample)
        mask = torch.LongTensor(mask)
        label = torch.LongTensor(label)

        return {
            'sample': sample, 
            'mask': mask, 
            'label': label
        } 