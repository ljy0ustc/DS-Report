import torch
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score
import wandb
def train(args,train_dataloader,valid_dataloader,model):
    lr_bert = args.lr_bert
    lr_main = args.lr_main
    num_epoch = args.num_epochs
    patience = args.patience
    cre = nn.BCELoss(reduction='mean')
    bert_param=[]
    main_param=[]
    # wandb.init(
    #         # set the wandb project where this run will be logged
    #         project="Fake Account Detection"
    #     )
    model=model.cuda()
    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'bert' in name:
                bert_param.append(p)
            else: 
                main_param.append(p)
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)  # xavier的初始化方式
        
    
    optimizer_main_group = [
            {'params': bert_param, 'weight_decay': args.weight_decay_bert, 'lr': lr_bert},
            {'params': main_param, 'weight_decay': args.weight_decay_main, 'lr': lr_main}
        ]

    optimizer = torch.optim.Adam(optimizer_main_group)  # 优化器
    scheduler = StepLR(optimizer, step_size=args.when,gamma=0.1)  # 学习率衰减
    train_losses = []
    train_acces = []
    train_f1s=[]
    eval_losses=[]
    eval_acces = []
    eval_f1s = []
    best_f1 = 0.0
    #训练
    for epoch in range(num_epoch):

        print(f"——————epoch {epoch}:——————")

        # 训练开始
        model.train()
        train_loss = 0
        train_acc = 0
        train_f1 = 0
        for batch in tqdm(train_dataloader, desc='training'):
            bert_sentences, bert_sentence_types, bert_att_mask, label = batch
            bert_sentences,bert_sentence_types, bert_att_mask, label = bert_sentences.cuda(),bert_sentence_types.cuda(), bert_att_mask.cuda(), label.cuda()

            output = model(bert_sentences,bert_sentence_types, bert_att_mask)

            Loss = cre(output, label)
          
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            pre = output.clone()
            pre[output>=0.5]=1.
            pre[output<0.5]=0.
            y_true = label.view(-1).cpu().detach().numpy()
            y_pre = pre.view(-1).cpu().detach().numpy()
            acc = accuracy_score(y_true=y_true,y_pred=y_pre)
            f1 = f1_score(y_true=y_true,y_pred=y_pre,average='weighted')
            
            train_loss+=Loss
            train_acc += acc
            train_f1 += f1
        scheduler.step()
        print(f'train loss: {train_loss}')
        print(f'train_acc: {train_acc / len(train_dataloader)}')
        print(f'train_f1: {train_f1 / len(train_dataloader)}')
        # wandb.log({"Train/Loss": train_loss/len(train_dataloader), 
        #                "Train/ACC": train_acc/len(train_dataloader),
        #                'Train/F1': train_f1/len(train_dataloader)
        #                })
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(train_loss)
        train_f1s.append(train_f1 / len(train_dataloader))

        model.eval()
        eval_loss = 0
        eval_acc = 0
        eval_f1 = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                feature, label = batch
                feature, label = feature.cuda(), label.cuda()
            # output = model(bert_sentences,bert_sentence_types, bert_att_mask)
                output = model(feature)
                Loss = cre(output, label)
                pre = output.clone()
                pre[output>=0.5]=1.
                pre[output<0.5]=0.
                y_true = label.view(-1).cpu().detach().numpy()
                y_pre = pre.view(-1).cpu().detach().numpy()
                acc = accuracy_score(y_true=y_true,y_pred=y_pre)
                f1 = f1_score(y_true=y_true,y_pred=y_pre,average='weighted')
                eval_loss += Loss
                eval_acc += acc
                eval_f1 += f1

            eval_loss = eval_loss / (len(valid_dataloader))
            eval_acc = eval_acc / (len(valid_dataloader))
            eval_f1 = eval_f1 / (len(valid_dataloader))
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                torch.save(model.state_dict(),'best_acc.pth')
            else:
                patience = patience-1
            eval_losses.append(eval_loss)
            eval_acces.append(eval_acc)
            eval_f1s.append(eval_f1)
            # wandb.log({"Valid/Loss": eval_loss, 
            #            "Valid/ACC": eval_acc,
            #            'Valid/F1': eval_f1
            #            })
            print("eval loss: {}".format(eval_loss))
            print("eval accuracy: {}".format(eval_acc))
            print("eval f1: {}".format(eval_f1))
            

            if patience==0:
                break
    return train_losses, train_acces, eval_acces, eval_losses, best_f1
