
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from model.contextnet.layer import Swish, depthwise_separable_1d_cnn_layer, SELayer

class ContextBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_layers,
                 kernel_size,
                 stride,
                 padding,
                 residual=True):
        super(ContextBlock, self).__init__()

        
        self.residual = residual
        self.num_layers = num_layers
        
        self.selayer = SELayer(out_channels)

        self.convlayer = depthwise_separable_1d_cnn_layer(in_channels,
                                                          out_channels,
                                                          kernel_size,
                                                          padding,
                                                          stride
                                                          )
        
        self.residual_conv = depthwise_separable_1d_cnn_layer(in_channels,
                                                          out_channels,
                                                          kernel_size,
                                                          padding,
                                                          stride,
                                                          activation = False)
        
        
        if stride == 1:
            self.blocks = nn.ModuleList([depthwise_separable_1d_cnn_layer(in_channels,
                                                                        out_channels,
                                                                        kernel_size,
                                                                        padding,
                                                                        stride = 1
                                                                        ) for block_id in range(self.num_layers)])
        elif stride == 2:
            self.blocks = nn.ModuleList([depthwise_separable_1d_cnn_layer(in_channels,
                                                                        out_channels,
                                                                        kernel_size,
                                                                        padding,
                                                                        stride = 1 if block_id < 4 else 2
                                                                        ) for block_id in range(self.num_layers)])    
        self.swish =Swish()
        
    def forward(self, x, x_len):
        
        output = x
        output_lengths = x_len
        
        for block in self.blocks:
            output, output_lengths = block(output, output_lengths)
        
        output = self.selayer(output)
        
        if self.residual:
            residual, _ = self.residual_conv(x, x_len)
            output += residual
            
        return self.swish(output), output_lengths
    
    def make_conv_blocks(input_dim, alpha):
        
        
        conv_blocks = nn.ModuleList()
        
        #conv_blocks.append(nn.Conv1d(input_dim, 256 * alpha, kernel_size=1))
        
        
        
        # C0
        conv_blocks.append(ContextBlock(in_channels = input_dim, out_channels = 256 * alpha, num_layers = 1,
                                        kernel_size = 5, stride = 1, padding = 0, residual=False))
        # C1-C2
        conv_blocks.append(ContextBlock(in_channels = 256 * alpha , out_channels = 256 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 256 * alpha , out_channels = 256 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        
        
        # C3 
        conv_blocks.append(ContextBlock(in_channels = 256 * alpha , out_channels = 256 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 2, padding = 0, residual=True))
        
        
        # C4 - C6
        conv_blocks.append(ContextBlock(in_channels = 256 * alpha , out_channels = 256 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 256 * alpha , out_channels = 256 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 256 * alpha , out_channels = 256 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        
        
        # C7
        conv_blocks.append(ContextBlock(in_channels = 256 * alpha , out_channels = 256 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 2, padding = 0, residual=True))
        # C8 ~ 10
        conv_blocks.append(ContextBlock(in_channels = 256 * alpha , out_channels = 256 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 256 * alpha , out_channels = 256 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 256 * alpha , out_channels = 256 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        
        '''
        # channel up
        conv_blocks.append(ContextBlock(in_channels = 256 * alpha , out_channels = 512 * alpha, num_layers = 1,
                                        kernel_size = 1, stride = 1, padding = 0, residual=False))
        
        # C11 ~ C13
        conv_blocks.append(ContextBlock(in_channels = 512 * alpha , out_channels = 512 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 512 * alpha , out_channels = 512 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 512 * alpha , out_channels = 512 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        
        # C14
        conv_blocks.append(ContextBlock(in_channels = 512 * alpha , out_channels = 512 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 2, padding = 0, residual=True))
        
        # C15 ~ C21
        conv_blocks.append(ContextBlock(in_channels = 512 * alpha , out_channels = 512 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 512 * alpha , out_channels = 512 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 512 * alpha , out_channels = 512 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 512 * alpha , out_channels = 512 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 512 * alpha , out_channels = 512 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 512 * alpha , out_channels = 512 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        conv_blocks.append(ContextBlock(in_channels = 512 * alpha , out_channels = 512 * alpha, num_layers = 5,
                                        kernel_size = 5, stride = 1, padding = 0, residual=True))
        
        # channel up
        conv_blocks.append(ContextBlock(in_channels = 512 * alpha , out_channels = 640 * alpha, num_layers = 1,
                                        kernel_size = 1, stride = 1, padding = 0, residual=False))
        
        # C22
        conv_blocks.append(ContextBlock(in_channels = 640 * alpha, out_channels = 640 * alpha, num_layers = 1,
                                        kernel_size = 5, stride = 1, padding = 0, residual=False))
        
        '''
        return conv_blocks
