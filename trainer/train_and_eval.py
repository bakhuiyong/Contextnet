import torch
import torch.nn as nn
from tqdm import tqdm
import os

from utils.metrics import label_to_string
from utils import chk_path, logger, epoch_log
from trainer.checkpoint import save
import jiwer

def train_and_eval(epochs, train_model, model, optimizer, schedules,criterion, train_dataloader, valid_dataloader, device):
    best_loss = 10101.0
    best_cer = 10101.0
    
    
    optimizer.zero_grad()
    
    logger.info("checkpoint saves in {} directory".format(chk_path))
    os.makedirs(chk_path, exist_ok=True)
    
    
    for epoch in range(0, epochs):
        train_loss  = train(train_model, optimizer, schedules,criterion, train_dataloader, device)
        val_cer  = valid(model, valid_dataloader, device)
        
        if best_loss>train_loss:
            best_loss = train_loss
               
        if best_cer>( val_cer*100 ):
            best_cer = ( val_cer*100 )
            save(os.path.join(chk_path, 'best_cer.pth'), epoch, model, optimizer)
          
        logger.info(epoch_log.format("info", epoch, train_loss, val_cer))
               

def train(model, optimizer, schedules,criterion, dataloader,device):
    losses = 0
    model.train()
    epoch_iterator = tqdm(dataloader, total=None)
    
    mixed_precision = True
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    accumulated_steps = 2
    acc_step = 0
    
    for step, (batch) in enumerate(epoch_iterator):
        
        batch = [elt.to(device) for elt in batch]

        inputs, targets, input_length, target_length = batch
        
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            logit, logit_length = model(inputs, input_length)
            loss_mini = criterion(logit, targets, logit_length, target_length)
            loss = loss_mini / accumulated_steps
        
        scaler.scale(loss).backward()
        
        losses += loss.detach()
        acc_step += 1
        
        if acc_step < accumulated_steps:
            continue
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        schedules.step()
        acc_step = 0
        
        epoch_iterator.set_description("model step: {} - mean loss {:.4f} - batch loss: {:.4f} - learning rate: {:.6f}".
                                       format(schedules.model_step, losses / (step + 1), loss, optimizer.param_groups[0]['lr']))

        
           
    return ( losses/ ( step+1 ) )

def valid(model, dataloader, device):
    
    total_cer = 0
    step = 0
    model.eval()
    
    eval_iterator = tqdm(dataloader, total=None)
    
    for step, batch in enumerate(eval_iterator):
            
        batch = [elt.to(device) for elt in batch]
            
        inputs, _, input_length, _ = batch
            
        with torch.no_grad():
                
            outputs_pred = model.gready_search_decoding(inputs, input_length)
            
        outputs_true = label_to_string(torch.tensor(batch[1].tolist()[0]))

            
        batch_cer = jiwer.cer(outputs_true.replace(' ',''), outputs_pred[0].replace(' ',''))
        total_cer += batch_cer
    
    return (total_cer/ (step+1))