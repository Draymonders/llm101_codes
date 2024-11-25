import torch.nn as nn
from transformers import BertModel


class NewsClassifier(nn.Module):
    def __init__(self, n_classes, model_path):
        super(NewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        return self.fc(output)

    def get_embeddings(self, input_ids, attention_mask):
        """Get BERT last layer [CLS] vector as text representation"""
        outputs = self.bert(input_ids, attention_mask)
        return outputs.last_hidden_state[:, 0, :]
