import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(nn.Conv1d(channels, channels, kernel_size, 
                                        padding=(kernel_size-1)*d//2, dilation=d))

    def forward(self, x):
        for conv in self.convs:
            h = F.leaky_relu(x, 0.1)
            h = conv(h)
            x = x + h
        return x


class HiFiGANGenerator(nn.Module):    
    def __init__(
        self,
        num_mels=80,
        channels=256,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates=(8, 8, 2, 2),
        upsample_initial_channel=512,
    ):
        super().__init__()
        
        self.num_mels = num_mels
        self.channels = channels
        self.conv_pre = nn.Conv1d(num_mels, channels, 7, 1, padding=3)

        self.upsamples = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        
        for i, (rate, kernel) in enumerate(zip(upsample_rates, resblock_kernel_sizes)):
            upsample_layer = nn.ConvTranspose1d(
                channels, channels, 
                kernel_size=rate * 2, 
                stride=rate, 
                padding=rate // 2
            )
            self.upsamples.append(upsample_layer)
            
            for dilation_size in resblock_dilation_sizes[i]:
                res_block = ResBlock(channels, kernel, [dilation_size])
                self.resblocks.append(res_block)
        
        self.conv_post = nn.Conv1d(channels, 1, 7, 1, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, mel):
        x = self.conv_pre(mel)
        
        for i, upsample in enumerate(self.upsamples):
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)
            resblock_idx = i * len([1, 1, 1])
            for j in range(resblock_idx, min(resblock_idx + 3, len(self.resblocks))):
                x = self.resblocks[j](x)
        
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = self.tanh(x)
        
        return x

    def inference(self, mel):
        return self.forward(mel)


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for p in periods:
            self.discriminators.append(PeriodDiscriminator(p))

    def forward(self, x):
        results = []
        for disc in self.discriminators:
            results.append(disc(x))
        return results


class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (3, 9), padding=(1, 4)),
            nn.Conv2d(32, 64, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(64, 128, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(128, 256, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(256, 512, (3, 3), padding=(1, 1)),
        ])
        self.conv_post = nn.Conv2d(512, 1, (3, 3), padding=(1, 1))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "constant", 0)
            t = t + n_pad
        
        x = x.reshape(b, c, t // self.period, self.period)
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
        
        x = self.conv_post(x)
        
        return x
