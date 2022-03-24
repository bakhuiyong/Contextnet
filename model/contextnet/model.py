import torch
import torch.nn as nn
from ctcdecode import CTCBeamDecoder
from utils.convert import index2char
# Modules

from model.contextnet.encoder import (
    ContextnetEncoder
)
from utils.metrics import index2char, EOS_token, SOS_token, PAD_token

class contextnet(nn.Module):
    def __init__(
        self, 
        out_dim
    ):
        super(contextnet,self).__init__()
        
        
        self.init_type = "xavier_uniform"

        
        
        self.encoder = ContextnetEncoder()
        
        self.fc = nn.Linear(256, out_dim)
        

    def forward(self, inputs, input_length):
        
        enc_out, enc_out_len = self.encoder(inputs,input_length)
        
        preds = self.fc(enc_out)    
        
        return preds, enc_out_len
    
    def gready_search_decoding(self, x, x_len):
    

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len = self.encoder(x, x_len)[:2]

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        # Softmax -> Log > Argmax -> (B, T)
        preds = logits.log_softmax(dim=-1).argmax(dim=-1)

        # Batch Pred List
        batch_pred_list = []

        # Batch loop
        for b in range(logits.size(0)):

            # Blank
            blank = False

            # Pred List
            pred_list = []

            # Decoding Loop
            for t in range(logits_len[b]):
                # Blank Prediction
                if preds[b, t] == 0:
                    blank = True
                    continue

                # First Prediction
                if len(pred_list) == 0:
                    pred_list.append(preds[b, t].item())

                # New Prediction
                elif pred_list[-1] != preds[b, t] or blank: 
                    pred_list.append(preds[b, t].item())
                
                # Update Blank
                blank = False
            pred_list = self.convert(pred_list) 
            batch_pred_list.append(pred_list)
        # Decode Sequences
        return batch_pred_list
    
    def convert(self,labels):
        sent = str()
        for i in labels:
            if i == EOS_token:
                break
            if i==SOS_token or i==PAD_token:
                continue
            sent += index2char[i]
        return sent
    
    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
    
    def beam_search_decoding(self, x, x_len, beam_size=None):
    
        import json
        with open('/home/dkdlenrh/contextnet_ctc/data/kor_syllable.json', 'r') as f:
            vocab = json.load(f) 
        id2char = list(vocab)
        
        decoder = CTCBeamDecoder(
            id2char,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=len(id2char),
            cutoff_prob=1.0,
            beam_width=beam_size,
            num_processes=8,
            blank_id=0,
            log_probs_input=True
        )

        # index2char, char2index
        
        
        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len = self.encoder(x, x_len)[:2]

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        # Softmax -> Log
        logP = logits.softmax(dim=-1).log()

        # Beam Search Decoding
        
        beam_results, beam_scores, timesteps, out_lens = decoder.decode(logP, logits_len)

        # Batch Pred List
        batch_pred_list = []

        # Batch loop
        for b in range(logits.size(0)):
            batch_pred_list.append(beam_results[b][0][:out_lens[b][0]].tolist())

        
        
        batch_pred_list = batch_pred_list[0]
        batch_pred_list = self.convert(batch_pred_list)
        return batch_pred_list