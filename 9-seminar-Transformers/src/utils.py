# Created by: c00k1ez (https://github.com/c00k1ez)

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import random
import numpy as np 


def get_answer(text: str, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
    cntx_token_id, answer_token_id = tokenizer.additional_special_tokens_ids
    context = tokenizer.encode(text)
    context = [tokenizer.bos_token_id] + [cntx_token_id] + context + [answer_token_id]
    context = torch.LongTensor([context])
    ans = model.generate(input_ids=context, max_length=100, temperature=0.7)[0][1:-1]
    return tokenizer.decode(ans)


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
