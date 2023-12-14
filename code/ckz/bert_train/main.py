import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU
from dataloader import getloader
from train import train
from model import LanguageEmbedding
from config import *
def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor') 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  
        torch.set_default_tensor_type('torch.cuda.FloatTensor') 
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False  

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    batch_size = args.batch_size
    train_loader = getloader(mode='train', batch_size=batch_size, shuffle=True)
    print('Training data loaded!')
    valid_loader = getloader(mode='valid', batch_size=batch_size, shuffle=False)
    print('Validation data loaded!')
    print('Finish loading the data....')

    # torch.autograd.set_detect_anomaly(True)
    model = LanguageEmbedding(in_size=18,hidden_size=18,dropout=0.2)

    train_losses, train_acces, eval_acces, eval_losses, best_f1=train(args,train_dataloader=train_loader,valid_dataloader=valid_loader,model=model)
    print(f'best f1-score on dev set:{best_f1}')