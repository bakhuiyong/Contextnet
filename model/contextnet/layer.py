
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class Swish(nn.Module):
    
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs):
        return inputs * inputs.sigmoid()

class depthwise_separable_1d_cnn_layer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 padding,
                 stride,
                 activation = True):
        
        super(depthwise_separable_1d_cnn_layer, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.activation = activation
        self.swish = Swish()
        
        self.layer = nn.Sequential(
                                    nn.Conv1d(in_channels, 
                                              in_channels, 
                                              kernel_size=self.kernel_size, 
                                              stride = self.stride,
                                              dilation=1,
                                              padding = (self.kernel_size - 1) // 2 if self.stride==1 else self.padding,
                                              groups = in_channels),
                                    nn.Conv1d(in_channels, 
                                              out_channels, 
                                              kernel_size=1),
                                    nn.BatchNorm1d(num_features=out_channels))


    def forward(self, x, x_len):
        
        if x_len is not None:
          
            if self.stride == 1:
                x_len = torch.div(x_len - 1, self.stride, rounding_mode='floor') + 1
            else:
                x_len = torch.div(x_len + 2 * self.padding - 1 * (self.kernel_size - 1) -1, self.stride, rounding_mode='floor') + 1
            
        x = self.layer(x)
        
        if self.activation:
            x = self.swish(x)
        
        return x, x_len
  

class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 8, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        b, c, _ = x.size()
        
        y = self.avg_pool(x).view(b, c)
    
        y = self.fc(y).view(b, c, 1)
        
        return x * y.expand_as(x) 
    

