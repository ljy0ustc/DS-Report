import requests
import json
import os

def get_img(url,path):
    if url:
        ps=url.split(".")[-1]
        ps=ps.lower()
        if ps in ["jpg","png","jpeg","gif"]:
            print(url)
            response = requests.get(url)
            with open(path+"."+ps, 'wb') as fw:
                fw.write(response.content)

def get_img2(url,path):
    if url:
        print(url)
        response = requests.get(url)
        with open(path+".jpg", 'wb') as fw:
            fw.write(response.content)

raw_data_dir = './ref/raw/'
processed_data_dir = './ref/processed/'
for file_name in ['test']:
    with open(os.path.join(raw_data_dir,file_name+".json"), 'r') as fr:
        data = json.load(fr)
        print(len(data))
        for each_data in data:
            id_str=each_data["user"]["id_str"]
            if "profile_image_url" in each_data["user"]:
                profile_image_url=each_data["user"]["profile_image_url"]
                get_img(profile_image_url,os.path.join(processed_data_dir+"profile_image",id_str))
            if "profile_banner_url" in each_data["user"]:
                profile_banner_url=each_data["user"]["profile_banner_url"]
                get_img2(profile_banner_url,os.path.join(processed_data_dir+"profile_banner",id_str))
            #theme=int(profile_background_image_url.split("/")[-2].split("theme")[-1])
            #profile_background_theme_dict[id_str]=theme

#with open(os.path.join(processed_data_dir,'profile_background_theme.json'), 'w') as fw:
#    json.dump(profile_background_theme_dict, fw)