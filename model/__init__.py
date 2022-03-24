import torch
from torch import nn
import os

def build_model(conf):
    model_type = conf['setting']['model_type']
    if model_type=='context_ctc':
        from model.contextnet.model import contextnet as model
   
    return model(**conf['model'])