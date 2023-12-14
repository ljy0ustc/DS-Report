import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
import json
import os
from PIL import Image

from torchvision import transforms


class CompData(data.Dataset):
    def __init__(self, data_dir=r'data/ref',
                 class_num=2,
                 stage=None):
        # Set all input args as attributes
        self.__dict__.update(locals())

        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 
        raw_file_path = op.join(op.join(self.data_dir, 'raw'), self.stage+".json")
        processed_file_path = op.join(op.join(self.data_dir, 'processed'), self.stage+".json")
        self.processed_file_dir = op.join(self.data_dir, 'processed')
        #with open(raw_file_path, 'r') as f:
        with open(processed_file_path, 'r') as f:
            self.data = json.load(f)
        self.profile_image_file_name_list=os.listdir(op.join(self.processed_file_dir,"profile_image"))
        self.pro_profile_image_file_name_list=[i.split(".")[0] for i in self.profile_image_file_name_list]
        self.profile_banner_file_name_list=os.listdir(op.join(self.processed_file_dir,"profile_banner"))
        self.pro_profile_banner_file_name_list=[i.split(".")[0] for i in self.profile_banner_file_name_list]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ViT通常需要224x224的输入
            transforms.ToTensor(),  # 将PIL图像转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化，这些值是预训练模型的训练数据的均值和标准差
        ])
        self.num_feature_list=['followers_count','friends_count','listed_count','statuses_count','favourites_count']
        self.bool_feature_list=['geo_enabled','verified','contributors_enabled','is_translator','is_translation_enabled','protected','profile_use_background_image','has_extended_profile','default_profile','default_profile_image','following','follow_request_sent','notifications']

    def __len__(self):
        return len(self.data)

    def to_one_hot(self, idx):
        out = np.zeros(self.class_num, dtype=float)
        out[idx] = 1
        return out

    def __getitem__(self, idx):
        each_data=self.data[idx]
        id_str=each_data["user"]["id_str"]
        if "profile_background_image_url" in each_data["user"]:
            profile_background_image=each_data["user"]["profile_background_image_url"]
            if profile_background_image:
                theme=int(profile_background_image.split("/")[-2].split("theme")[-1])
            else:
                theme=0
        else:
            theme=0
        if id_str in self.pro_profile_image_file_name_list:
            #img_path=op.join(op.join(self.processed_file_dir,"profile_image"),self.profile_image_file_name_list[self.pro_profile_image_file_name_list.index(id_str)])
            #try:
            #    profile_image=Image.open(img_path)
            #    print(img_path)
            #except:
            #    profile_image=Image.new("RGB", (48, 48))
            #profile_image = self.transform(profile_image)
            #profile_image = profile_image.unsqueeze(0)
            profile_image=op.join(op.join(self.processed_file_dir,"profile_image"),self.profile_image_file_name_list[self.pro_profile_image_file_name_list.index(id_str)])
        else:
            #profile_image=Image.new("RGB", (48, 48))
            profile_image=''
        if id_str in self.pro_profile_banner_file_name_list:
            #img_path=op.join(op.join(self.processed_file_dir,"profile_banner"),self.profile_banner_file_name_list[self.pro_profile_banner_file_name_list.index(id_str)])
            #try:
            #    profile_banner=Image.open(img_path)
            #    print(img_path)
            #except:
            #    profile_banner=Image.new("RGB", (128, 128))
            #profile_banner = self.transform(profile_banner)
            #profile_banner = profile_banner.unsqueeze(0)
            profile_banner=op.join(op.join(self.processed_file_dir,"profile_banner"),self.profile_banner_file_name_list[self.pro_profile_banner_file_name_list.index(id_str)])
        else:
            #profile_banner=Image.new("RGB", (128, 128))
            profile_banner=''
        if each_data["label"] == 'bot':
            label=self.to_one_hot(1)
        else:
            label=self.to_one_hot(0)
        num_features=[]
        bool_features=[]
        for feature in self.num_feature_list:
            if feature in each_data["user"]:
                num_features.append(each_data["user"][feature])
            else:
                num_features.append(0)
        for feature in self.bool_feature_list:
            if feature in each_data["user"]:
                bool_features.append(each_data["user"][feature])
            else:
                bool_features.append(0)
        return {
            'profile_image': profile_image,
            'profile_banner': profile_banner,
            'theme': theme,
            #'followers_count': each_data["user"]["followers_count"],
            #'friends_count': each_data["user"]["friends_count"],
            #'listed_count': each_data["user"]["listed_count"],
            #'favourites_count': each_data["user"]["favourites_count"],
            #'statuses_count': each_data["user"]["statuses_count"],
            'num_features': num_features,
            'bool_features': bool_features,
            'label': label
        }
            