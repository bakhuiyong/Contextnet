import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np



###############################################################################
# Audio Preprocessing
###############################################################################


class AudioPreprocessing(nn.Module):

    def __init__(self, sample_rate, n_fft, win_length_ms, hop_length_ms, n_mels, normalize, mean, std):
        super(AudioPreprocessing, self).__init__()
        self.win_length = int(sample_rate * win_length_ms) // 1000
        self.hop_length = int(sample_rate * hop_length_ms) // 1000
        self.Spectrogram = torchaudio.transforms.Spectrogram(n_fft, self.win_length, self.hop_length)
        self.MelScale = torchaudio.transforms.MelScale(n_mels, sample_rate, f_min=0, f_max=8000, n_stft=n_fft // 2 + 1)
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def forward(self, x, x_len):

        x = self.Spectrogram(x)

        x = self.MelScale(x)
        
        x = (x.float() + 1e-9).log().type(x.dtype)

        # Compute Sequence lengths 
        if x_len is not None:
            x_len = torch.div(x_len, self.hop_length, rounding_mode='floor') + 1

        # Normalize
        if self.normalize:
            x = (x - self.mean) / self.std

        return x, x_len

class SpecAugment(nn.Module):
    

    def __init__(self, spec_augment, mF, F, mT, pS):
        super(SpecAugment, self).__init__()
        self.spec_augment = spec_augment
        self.mF = mF
        self.F = F
        self.mT = mT
        self.pS = pS

    def forward(self, x, x_len):

        # Spec Augment
        if self.spec_augment:
        
            # Frequency Masking
            for _ in range(self.mF):
                x = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.F, iid_masks=False).forward(x)

            # Time Masking
            for b in range(x.size(0)):
                T = int(self.pS * x_len[b])
                
                for _ in range(self.mT):
                    x[b, :, :x_len[b]] = torchaudio.transforms.TimeMasking(time_mask_param=T).forward(x[b, :, :x_len[b]])

        return x
