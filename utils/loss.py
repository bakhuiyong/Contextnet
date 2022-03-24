import torch
from torch import nn

class CTCLOSS(nn.Module):
    def __init__(self, padding_idx):
        super(CTCLOSS, self).__init__()
        
        
        self.ctc = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
    

    def forward(self, pred, targets, preds_lengths, target_length):

        ctc_loss = self.ctc(log_probs = torch.nn.functional.log_softmax(pred, dim=-1).transpose(0, 1), 
                            targets = targets, 
                            input_lengths = preds_lengths, 
                            target_lengths = target_length)
        
        return ctc_loss
    
def build_criterion(conf):
    loss_type = conf['setting']['loss_type']
    if loss_type=='CTCLOSS':
        criterion = CTCLOSS(padding_idx=0).to('cuda')
        
    return criterion