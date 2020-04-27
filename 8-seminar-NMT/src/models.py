# Created by: c00k1ez (https://github.com/c00k1ez)

from typing import List

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


class GlobalAttention(nn.Module):
    def __init__(self):
        super(GlobalAttention, self).__init__()
    
    def forward(self, encoder_output, decoder_gru_output, encoder_mask, decoder_mask):
        # encoder_output shape : [batch_size, source_pad_len, hidden_size]
        # decoder_gru_output shape : [batch_size, target_pad_len, hidden_size]
        # encoder_mask shape : [batch_size, source_pad_len]
        # decoder_mask shape : [batch_size, target_pad_len]

        weighted_encoder = None
        attention_scores = None
        ################### INSERT YOUR CODE HERE ###################
        
        ################### INSERT YOUR CODE HERE ###################

        # weighted_encoder shape : [batch_size, target_pad_len, hidden_size]
        # attention_scores shape : [batch_size, target_pad_len, source_pad_len]      
        
        return weighted_encoder, attention_scores



class Decoder(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 hidden_size: int, 
                 embedding_size: int,
                 weight_tying: bool = True,
                 padding_idx: int = 3,
                 use_attention: bool = False):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_attention = use_attention

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.prehead = nn.Linear(hidden_size, embedding_size)
        self.head = nn.Linear(embedding_size, vocab_size)
        self.dropout = SpatialDropout()
        self.dropout_1 = nn.Dropout()
        self.attention = GlobalAttention()

        if weight_tying is True:
            self.head.weight = self.embedding.weight

    def forward(self, batch, hidden, encoder_outputs):
        target = batch['target_language_sentence']
        target_mask = batch['target_sentence_mask']
        encoder_mask = batch['source_sentence_mask']
        output = self.embedding(target) # [batch_size, target_pad_len, embedding_size]
        
        output, hidden = self.gru(output, hidden)
        # output shape: [batch_size, source_pad_len, hidden_size]
        # hidden shape: [1, batch_size, hidden_size]
        if self.use_attention:
          attn_output, attn_matrix = self.attention(encoder_outputs, output, encoder_mask, target_mask)
        # combine attention and GRU output! 
        output = self.dropout(output)
        output = self.prehead(output) # [batch_size, source_pad_len, embedding_size]
        output = self.head(self.dropout_1(output))
        return F.log_softmax(output, dim=2)



class NMTModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        source, source_mask = batch['source_sentence'], batch['source_sentence_mask']

        encoder_output, hidden = self.encoder((source, source_mask))
        decoder_output = self.decoder(batch, hidden, encoder_output)

        return decoder_output

    def translate(self, source_sentence: List[int], device) -> List[str]:
        self.eval()
        source_sentence = torch.LongTensor([source_sentence]).to(device)
        translated_sentence = [0, ]
        batch = {
            'source_sentence': source_sentence, 
            'source_sentence_mask': None,
            'target_language_sentence': None,
            'target_sentence_mask': None
        }
        with torch.no_grad():
            sentence_emb, hidden = self.encoder((source_sentence, None))

        while translated_sentence[-1] != 1:
            target = torch.LongTensor([translated_sentence]).type_as(source_sentence)
            batch['target_language_sentence'] = target
            with torch.no_grad():
                preds = self.decoder(batch, hidden, sentence_emb)

            token = torch.exp(preds).max(dim=2)[1][:,-1].cpu().item()
            translated_sentence.append(token)

        return translated_sentence



