import torch
from typing import List


class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, pad_len):
        self.data = data
        self.tokenizer = tokenizer
        self.pad_len = pad_len

    def __len__(self):
        return len(self.data)

    def _check_question_len_(self, question: List[str], max_len: int):
        if len(question) > max_len:
            question = question[:max_len]
        return question
    
    def __getitem__(self, indx):
        sample = self.data[indx]
        question1 = sample[0]
        question2 = sample[1]
        label = int(sample[2])
        question1 = self.tokenizer.tokenize(question1, add_special_tokens=False)
        question2 = self.tokenizer.tokenize(question2, add_special_tokens=False)
        # use first half of tokens for question1 and  another half for question2
        question1 = self._check_question_len_(question1, (self.pad_len - 2) // 2)
        question2 = self._check_question_len_(question2, (self.pad_len - 2) // 2)
        sample = ['[CLS]'] + question1 + ['[SEP]'] + question2
        attn_mask = [1] * len(sample) + [0] * (self.pad_len - len(sample))
        sample = sample + ['[PAD]'] * (self.pad_len - len(sample))
        sample = self.tokenizer.convert_tokens_to_ids(sample)
        assert len(sample) == len(attn_mask) == self.pad_len
        sample = torch.LongTensor(sample)
        attn_mask = torch.LongTensor(attn_mask)
        label = torch.LongTensor([label])
        return {
            'question_pairs': sample,
            'attention_mask': attn_mask,
            'label': label
        }
