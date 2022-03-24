import torch
import warnings
import argparse
warnings.filterwarnings('ignore')

from model import build_model
from utils import build_conf
from trainer import evaluate, load
from data.dataset import get_dataloader
import torch.optim as optim
from model.contextnet.schedules import (
    transformer_learning_rate_scheduler
)

def main(args):
    
    conf = build_conf(args.conf)
    
    batch_size = conf['train']['batch_size']
    
    test_dataloader = get_dataloader(conf['dataset']['valid'],
                                      conf['dataset']['root'],
                                      batch_size=batch_size, 
                                      mode='valid', 
                                      conf=conf)
    
    model = build_model(conf)
    
    optimizer = optim.Adam(model.parameters(),
                           lr=0, 
                           betas = (0.9, 0.98),
                           eps = 1e-9,
                           weight_decay=1e-6)

    K = 16
    saved_epoch = load(args, model, optimizer)
    eval_cer = evaluate(model, test_dataloader,K)
    
    print("eval_cer",eval_cer*100)

if __name__ == '__main__':

    print(torch.__version__)


    parser = argparse.ArgumentParser(description='End-to-End Speech Recognition Training')
    parser.add_argument('--conf', default='config/contextnet_ctc.yaml', type=str, help="configuration path for training")
    parser.add_argument('--load_model', default='/home/dkdlenrh/contextnet_ctc/checkpoint/01-30-07:31/best_cer.pth', type=str, help="evaluate from saved model")
    args = parser.parse_args()
    main(args)

# python evaluate.py --conf config/contextnet_ctc.yaml
