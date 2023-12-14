import json
import os
import numpy as np
import copy

raw_data_dir = './ref/raw/'
processed_data_dir = './ref/processed/'
data=[]
for file_name in ['train','dev','test']:
    with open(os.path.join(raw_data_dir,file_name+".json"),"r") as f:
        split_data=json.load(f)
        data+=split_data
print(len(data))

num_feature_list=['followers_count','friends_count','listed_count','statuses_count','favourites_count']
bool_feature_list=['geo_enabled','verified','contributors_enabled','is_translator','is_translation_enabled','protected','profile_use_background_image','has_extended_profile','default_profile','default_profile_image','following','follow_request_sent','notifications']
data_cal={}
for feature in num_feature_list:
    data_cal[feature]=[]
    for each_data in data:
        if feature in each_data["user"]:
            data_cal[feature].append(each_data["user"][feature])
statis={}
for feature in num_feature_list:
    t=np.array(data_cal[feature])
    #statis[feature]=(t.mean(),t.std())
    statis[feature]=(t.min(),t.max())
print(statis)

for file_name in ['train','dev','test']:
    with open(os.path.join(raw_data_dir,file_name+".json"),"r") as fr:
        data=json.load(fr)
        for each_data in data:
            for feature in num_feature_list:
                if feature in each_data["user"]:
                    each_data["user"][feature]=(each_data["user"][feature]-statis[feature][0])/(statis[feature][1]-statis[feature][0])
            for feature in bool_feature_list:
                if feature in each_data["user"]:
                    each_data["user"][feature]=int(each_data["user"][feature])
    with open(os.path.join(processed_data_dir,file_name+".json"),"w") as fw:
        json.dump(data,fw)