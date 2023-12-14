import json
import os
import copy

raw_data_dir = './ref/raw/'
output_data_path = './ref/test_result.txt'

with open(os.path.join(raw_data_dir,"test.json"),"r") as f:
    original_data=json.load(f)
result_data=copy.deepcopy(original_data)
with open(output_data_path,"r") as f:
    res=f.readlines()
    for i,res_ans in enumerate(res):
        result_data[i]["label"]=res_ans.strip()
with open(os.path.join(raw_data_dir,"test_result.json"),"w") as f:
    json.dump(result_data,f)