import argparse

def get_args():  # 获取参数
    parser = argparse.ArgumentParser(description='Fake Account Detection')
    parser.add_argument('-f', default='', type=str)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--lr_main', type=float, default=0.001,
                        help='initial learning rate for main model parameters')
    parser.add_argument('--lr_bert', type=float, default=1e-5,
                        help='initial learning rate for bert parameters')
    
    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_bert', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    
    # parser.add_argument('--optim', type=str, default='Adam',
    #                     help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=20, 
                        help='number of epochs (default: 20)')
    parser.add_argument('--when', type=int, default=10,
                        help='when to decay learning rate') 
    parser.add_argument('--patience', type=int, default=10,
                        help='when to stop training if best never change')
    args = parser.parse_args()
    return args