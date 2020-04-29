import torch


class MeanPooling(torch.nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, batch, mask=None):
        return batch.mean(dim=1)

class BertClassifier(torch.nn.Module):
    def __init__(self, bert_model):
        super(BertClassifier, self).__init__()
        self.bert_model = bert_model
        self.head = torch.nn.Linear(768, 2)
        self.dropout = torch.nn.Dropout()
        self.pooling = MeanPooling()
    
    def forward(self, batch):
        samples = batch['question_pairs']
        attn_mask = batch['attention_mask']
        embedding = self.bert_model(samples, attention_mask=attn_mask)[0]
        pooled = self.pooling(embedding)
        pooled = self.dropout(pooled)
        pooled = self.head(pooled)
        return pooled

