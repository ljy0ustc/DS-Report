import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

bert_path = '/data1/kezhou/pretrained_large_model/bert-base-uncased'

class predict_head(nn.Module):
    def __init__(self,in_size):
        super(predict_head, self).__init__()
        self.drop = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_size, 1)
        self.Sig = nn.Sigmoid()
    def forward(self, hidden_states):
        dropped = self.drop(hidden_states)
        logit = self.linear(dropped)
        y_pred = self.Sig(logit)
        return y_pred


class LanguageEmbedding(nn.Module):  # 使用bert编码
    def __init__(self):
        super(LanguageEmbedding, self).__init__()
        bertconfig = BertConfig.from_pretrained(bert_path, output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained(bert_path, config=bertconfig)
        self.predict_head = predict_head(in_size=768)

    def forward(self, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                attention_mask=bert_sent_mask,
                                token_type_ids=bert_sent_type)
        enc_word = bert_output[0]  # sequence representation
        r_text = enc_word[:,0,:] # (batch_size, emb_size), 把CLS作为整个句子的表征
        output = self.predict_head(r_text)
        return output   # 输出结果为0到1的数值。判断其为bot的概率。

