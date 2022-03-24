import torch
import warnings
import argparse
warnings.filterwarnings('ignore')

from model import build_model
from utils import build_conf
from trainer import train_and_eval, load
from utils.loss import build_criterion
from data.dataset import get_dataloader
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.contextnet.schedules import (
    transformer_learning_rate_scheduler
)


def main(args):
    
    conf = build_conf(args.conf)
    
    batch_size = conf['train']['batch_size']

    train_dataloader = get_dataloader(conf['dataset']['train'],
                                                     batch_size=batch_size, 
                                                     mode='train', 
                                                     conf=conf)
                    
    valid_dataloader = get_dataloader(conf['dataset']['valid'],
                                      batch_size=batch_size, 
                                      mode='valid', 
                                      conf=conf)

    
         
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(conf)
    model = model.to(device)
    
    train_model = nn.DataParallel(model)
    
    criterion = build_criterion(conf)
    
    
    optimizer = optim.Adam(model.parameters(),
                           lr=0, 
                           betas = (0.9, 0.98),
                           eps = 1e-9,
                           weight_decay=1e-6)

    schedules = transformer_learning_rate_scheduler(optimizer=optimizer, 
                                                    dim_model=conf['scheduler']['dim_model'], 
                                                    warmup_steps=conf['scheduler']['warmup_steps'], 
                                                    K=conf['scheduler']['k'])


    print("Number of parameters: %d" % model.get_param_size(model))

    

    train_and_eval(conf['train']['epochs'], 
                   train_model, 
                   model,
                   optimizer,
                   schedules, 
                   criterion, 
                   train_dataloader, 
                   valid_dataloader, 
                   device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-End Speech Recognition Training')
    parser.add_argument('--conf', default='config/contextnet_ctc.yaml', type=str, help="configuration path for training")
    args = parser.parse_args()
    main(args)


    # python train.py --conf config/contextnet_ctc.yaml