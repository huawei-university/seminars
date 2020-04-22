# Created by: c00k1ez (https://github.com/c00k1ez)

import torch
import torch.nn as nn
import torch.nn.functional as F 


class SpatialDropout(nn.Dropout2d):
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super(SpatialDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1)
        x = x.squeeze(2)
        
        return x


class Encoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embedding_size: int, 
                 hidden_size: int,
                 padding_idx: int = 3):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.dropout = SpatialDropout()

    def forward(self, batch, hidden=None):
        source, mask = batch
        embedded = self.embedding(source) # [batch_size, source_pad_len, embedding_size]
        
        output, hidden = self.gru(embedded, hidden) 
        # output shape: [batch_size, source_pad_len, hidden_size]
        # hidden shape: [1, batch_size, hidden_size]
        output = self.dropout(output)
        return output, hidden

    
class Decoder(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 hidden_size: int, 
                 embedding_size: int,
                 weight_tying: bool = True,
                 padding_idx: int = 3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.prehead = nn.Linear(hidden_size, embedding_size)
        self.head = nn.Linear(embedding_size, vocab_size)
        self.dropout = SpatialDropout()

        if weight_tying is True:
            self.head.weight = self.embedding.weight

    def forward(self, batch, hidden):
        target, mask = batch
        output = self.embedding(target) # [batch_size, target_pad_len, embedding_size]

        output, hidden = self.gru(output, hidden)
        # output shape: [batch_size, source_pad_len, hidden_size]
        # hidden shape: [1, batch_size, hidden_size]
        output = self.dropout(output)
        output = self.prehead(output) # [batch_size, source_pad_len, embedding_size]
        output = self.head(F.dropout(output))
        return F.log_softmax(output, dim=2)


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        source, source_mask = batch['source_sentence'], batch['source_sentence_mask']
        target, target_mask = batch['target_sentence'], batch['target_sentence_mask']

        encoder_output, hidden = self.encoder((source, source_mask))
        decoder_output = self.decoder((target, target_mask), hidden)

        return decoder_output

