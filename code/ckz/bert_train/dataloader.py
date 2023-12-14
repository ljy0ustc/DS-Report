import torch
from torch.utils.data import DataLoader, Dataset
import json
from transformers import *
f_train = open('/data1/kezhou/homework/src/dataset/train.json','r')
content_train = f_train.read()
f_dev = open('/data1/kezhou/homework/src/dataset/dev.json','r')
content_dev = f_dev.read()
bert_path = '/data1/kezhou/pretrained_large_model/bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)

class FDAdata(Dataset):  # Dataset
    def __init__(self,content):
        self.data = json.loads(content)  # list
    def __getitem__(self, idx):  
        name = self.data[idx]['user']['name']
        screen_name = self.data[idx]['user']['screen_name']
        description = self.data[idx]['user']['description']

        protected =  self.data[idx]['user']['protected']
        followers_count = self.data[idx]['user']['followers_count']
        friends_count = self.data[idx]['user']['friends_count']
        listed_count = self.data[idx]['user']['listed_count']
        favourites_count = self.data[idx]['user']['favourites_count']
        geo_enabled = self.data[idx]['user']['geo_enabled']
        verified = self.data[idx]['user']['verified']
        statuses_count = self.data[idx]['user']['statuses_count']
        contributors_enabled = self.data[idx]['user']['contributors_enabled']
        is_translator = self.data[idx]['user']['is_translator']
        is_translation_enabled = self.data[idx]['user']['is_translation_enabled']
        profile_background_tile = self.data[idx]['user']['profile_background_tile']
        profile_use_background_image = self.data[idx]['user']['profile_use_background_image']
        has_extended_profile = self.data[idx]['user']['has_extended_profile']
        default_profile = self.data[idx]['user']['default_profile']
        following = self.data[idx]['user']['following']
        follow_request_sent = self.data[idx]['user']['follow_request_sent']
        notifications = self.data[idx]['user']['notifications']
        translator_type = self.data[idx]['user']['translator_type']
        translator_type = False if translator_type =='none' else True
        label = self.data[idx]['label']
        label = 0 if label=='human' else 1  # 0代表human, 1代表bot
        return protected,followers_count,friends_count,listed_count,favourites_count,geo_enabled,verified,statuses_count,contributors_enabled,is_translator,\
        is_translation_enabled,profile_background_tile,profile_use_background_image,has_extended_profile,default_profile,following,follow_request_sent,\
        notifications,translator_type,label
    def __len__(self):
        return len(self.data)

def collate_fn(batch_data):
    label = []
    bert_details = []
    for i, sample in enumerate(batch_data):
        description,data_label=sample
        text = " ".join(description)
        encoded_bert_sent = bert_tokenizer.encode_plus(
                    text, max_length=50, add_special_tokens=True, truncation=True, padding='max_length', pad_to_max_length=True)
        bert_details.append(encoded_bert_sent)
        label.append(data_label)
    bert_sentences = torch.LongTensor([sample['input_ids'] for sample in bert_details])
    bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
    bert_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])
    label = torch.Tensor(label).reshape(-1,1)
    return bert_sentences, bert_sentence_types, bert_att_mask, label 

def getloader(mode,batch_size,shuffle):  # mode决定数据集类型. 本函数返回dataloader.
    content = content_train if mode == 'train' else content_dev
    dataset = FDAdata(content)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)
    return data_loader
