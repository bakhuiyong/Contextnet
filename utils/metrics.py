import Levenshtein as Lev
import torch

from utils.convert import char2index, index2char

SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = '_'
UNK_TOKEN = None

SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']

    
def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            if i.item()==2001 or i.item()==0 or i.item()==None:
                continue
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                if j.item()==2001 or j.item()==0 or j.item()==None:
                    continue
                sent += index2char[j.item()]
            sents.append(sent)

        return sents
