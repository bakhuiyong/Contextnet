import torch
import torchaudio

import os
from utils.convert import char2index

SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = '_'
UNK_TOKEN = None

import re
def normalize(text): #텍스트 클리닝
    text = text.upper()
    text = re.sub('[^A-Z0-9가-힣-\s]', '', text) 
    return text

class dataset_load(torch.utils.data.Dataset): 
    def __init__(self, trn):
    
        with open(trn, 'r') as f:
            self.data = f.read().strip().split('\n')

        self.prep_data(500)


    def __getitem__(self, i):

        fname, script = self.data[i]
        script = normalize(script)
        seq = self.scr_to_seq(script)
        
        return [torchaudio.load(fname)[0], torch.tensor(seq)]

    def __len__(self):

        return len(self.data)
    
    def prep_data(self, tgt_max_len=None, symbol=' :: '):
        """ 
        Data preparations.
        (A, B): Tuple => A: Audio file path, B: Transcript
        """
        temp = []
        #print("self.data",self.data)
        for line in self.data:
            
            fname, script = line.split(' :: ') 
            script = script.replace('[UNK] ','')
            
            if tgt_max_len is not None:
                if len(script)>tgt_max_len:
                    continue
            temp.append((fname,script))
        self.data = temp
        
    def scr_to_seq(self, scr):
        seq = list()
        for c in scr:
            if c in char2index:
                seq.append(char2index.get(c))
            else:
                if UNK_TOKEN is not None:
                    seq.append(char2index.get(UNK_TOKEN))
                else:
                    continue
        return seq  
    
def collate_fn_pad(batch):
    
    # Sorting sequences by lengths
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)

        # Pad data sequences
    data = [item[0].squeeze() for item in sorted_batch]
    data_lengths = torch.tensor([len(d) for d in data],dtype=torch.long) 
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
        # Pad labels
    target = [item[1] for item in sorted_batch]
    target_lengths = torch.tensor([t.size(0) for t in target],dtype=torch.long)
    target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=0)

    return data, target, data_lengths, target_lengths

def get_dataloader(trn, batch_size=16, mode='valid', conf=None):
    
    dataset = dataset_load(trn)

    if mode == 'train':
        dataset = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=8, 
                                                    collate_fn=collate_fn_pad, 
                                                    drop_last=True, 
                                                    sampler=None, 
                                                    pin_memory=False)
        
         
    else:
        dataset =  torch.utils.data.DataLoader(dataset, 
                                               batch_size=1, 
                                               shuffle=False, 
                                               num_workers=8, 
                                               collate_fn=collate_fn_pad, 
                                               sampler=None, 
                                               pin_memory=False)
    

    return dataset