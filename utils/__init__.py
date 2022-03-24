import datetime
import os
import yaml
import logging

def build_conf(conf_path='config/contextnet_ctc.yaml'):
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)
    from utils.convert import voca_size
    conf['model']['out_dim'] = voca_size
    
    return conf

def get_now():
    now = datetime.datetime.now()
    cur = now.strftime('%m-%d-%H:%M')
    
    return cur

def make_chk(root='checkpoint'):
    cur = get_now()
    path = os.path.join(root,cur)
    
    return path, cur

chk_path, cur = make_chk()

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('log/{}.log'.format(cur))

logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.setLevel(level=logging.DEBUG)

train_log = "[{}] epoch: {} loss : {:.4f} cer: {:.2f}"
valid_log = "[{}] epoch: {} cer: {:.2f}"

eval_log = "[{}] cer: {:.2f}"

epoch_log = "[{}] {} epoch is over, train_avg_loss {:.2f}, vali_avg_cer {:.4f}"   

cer_log = "ground_truth : {}\nhypothesis : {}\nCer : {:.3f}%, Corr : {}, Sub : {}, Ins : {}, Del : {}\n\n"