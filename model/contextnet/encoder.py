# PyTorch
import torch
import torch.nn as nn

# Blocks
from model.contextnet.block import (
    ContextBlock
)

from model.contextnet.modules import (
    AudioPreprocessing,
    SpecAugment
)

class ContextnetEncoder(nn.Module):
    
    def __init__(self):
        super(ContextnetEncoder, self).__init__()
        
        self.dropout = nn.Dropout(0.1)

        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length_ms = 25
        self.hop_length_ms = 10
        self.n_mels = 80
        self.normalize = False
        self.preprocessing = AudioPreprocessing(self.sample_rate, 
                                                self.n_fft, 
                                                self.win_length_ms, 
                                                self.hop_length_ms, 
                                                self.n_mels , 
                                                self.normalize, 
                                                None, 
                                                None)
        
        
        self.augment = SpecAugment(True, 2, 27, 2, 0.25)
        
        self.blocks = ContextBlock.make_conv_blocks(self.n_mels, 1)
        
        
        
        
        
    def forward(self, x, x_len=None):

        # Audio Preprocessing
        x, x_len = self.preprocessing(x, x_len)

        '''
        if self.training:
            x = self.augment(x, x_len)
        '''
        
        for block in self.blocks:
            x, x_len = block(x, x_len)
        
        x = x.transpose(1, 2)
        
        return x, x_len