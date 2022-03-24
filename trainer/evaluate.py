import torch
import torch.nn as nn
from tqdm import tqdm

from utils import logger, cer_log
from utils.metrics import label_to_string
import jiwer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, dataloader,k):
    model.eval()
    total_cer = 0
    eval_iterator = tqdm(dataloader, total=None)
    
    for step, batch in enumerate(eval_iterator):
        
        #batch = [elt.to(device) for elt in batch]
        inputs, targets, input_length, target_length = batch
        
        with torch.no_grad():   
            if k==1: 
                outputs_pred = model.gready_search_decoding(inputs, input_length)
                outputs_true = label_to_string(torch.tensor(batch[1].tolist()[0]))
                error = jiwer.cer(outputs_true.replace(' ',''), outputs_pred[0].replace(' ',''),return_dict=True)
                
                logger.info(cer_log.format(outputs_true,
                    outputs_pred[0],
                    error['cer']*100,
                    error['hits'],
                    error['substitutions'],
                    error['insertions'],
                    error['deletions']))
                
                
            else:
                outputs_pred = model.beam_search_decoding(inputs, input_length, k)
                outputs_true = label_to_string(torch.tensor(batch[1].tolist()[0]))
                error = jiwer.cer(outputs_true.replace(' ',''), outputs_pred.replace(' ',''),return_dict=True)
        
                logger.info(cer_log.format(outputs_true,
                    outputs_pred,
                    error['cer']*100,
                    error['hits'],
                    error['substitutions'],
                    error['insertions'],
                    error['deletions']))
        
        
        total_cer += error['cer']
        
        
    return total_cer/(step+1)