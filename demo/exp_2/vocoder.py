import torch
import torch.nn as nn
import torch.nn.functional as F

class HiFiGANGenerator(nn.Module):
    def __init__(self, upsample_rates=[8,8,2,2], upsample_kernel_sizes=[16,16,4,4], 
                 resblock_kernel_sizes=[[3,7],[11,1],[3,7]], resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]],
                 num_mels=80):
        super().__init__()
        self.num_mels = num_mels
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.conv_pre = nn.Conv1d(num_mels, 512, 7, 1, 3)
        
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        
        for i, (r, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose1d(512 // (2**i), 512 // (2**(i+1)), k, r, (k-r+1)//2))
            
            ch = 512 // (2**(i+1))
            for rks, ds in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                res_block = nn.ModuleList()
                for j, (rk, d) in enumerate(zip(rks, ds)):
                    pad = (rk - 1) * d // 2
                    res_block.append(nn.Conv1d(ch, ch, rk, dilation=d, padding=pad))
                    res_block.append(nn.LeakyReLU(0.1))
                self.resblocks.append(res_block)
        
        self.conv_post = nn.Sequential(
            nn.Conv1d(512 // (2**len(upsample_rates)), 256, 7, 1, 3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 1, 7, 1, 3)
        )

    def forward(self, x):
        if x.dim() == 4:
            B, C, freq_bins, T = x.shape
            x = x.mean(dim=2)
        x = torch.log(torch.clamp(x, min=1e-5))
        x = self.conv_pre(x)
        
        res_idx = 0
        for i, up in enumerate(self.ups):
            x = F.leaky_relu_(up(x), 0.1)

            for _ in range(len(self.resblock_kernel_sizes)):
                res_out = x
                for layer in self.resblocks[res_idx]:
                    res_out = layer(res_out)
                x = res_out + x
                res_idx += 1
        
        x = self.conv_post(x)
        return torch.tanh(x)
