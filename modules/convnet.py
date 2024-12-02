import torch
import torch.nn as nn
import itertools


class StyleIN(nn.Module):
    def __init__(self, channels, spatial_dims, max_styles):
        super().__init__()
        self.mu = nn.Linear(1, channels)
        self.mu = nn.Parameter(torch.zeros([max_styles, channels]))
        self.sigma = nn.Parameter(torch.zeros([max_styles, channels]))
        self.sp_dims = spatial_dims

    def expand_dims(self, x):
        for i in self.sp_dims: x = x.unsqueeze(i)
        return x

    def forward(self, x, y):
        var, mean = torch.var_mean(x, dim=self.sp_dims, keepdim=True)
        n = (x - mean) * torch.rsqrt(var + 1e-6)
        return n * self.expand_dims(self.sigma[y]) + self.expand_dims(self.mu[y])


class PatchBlock(nn.Module):
    def __init__(self, input_channels, output_channels, strides, max_styles):
        super().__init__()
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              kernel_size=2*strides,
                              stride=strides,
                              padding=0)
        self.norm = StyleIN(input_channels, (2, 3), max_styles=max_styles)
        self.act = nn.ReLU()
        self.rezero = nn.Parameter(torch.Tensor([0.]))
        self.skip = nn.Identity() if input_channels == output_channels else nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
    def forward(self, x, y):
        res = self.conv(self.act(self.norm(x, y)))
        out = (res * self.rezero) + self.skip(self.pool(x))
        return out

class SequentialWithArgs(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input

class AutoConvertAttnBlock(nn.Module):
    def __init__(self, channels, num_heads, max_styles, recover_shape=True):
        super().__init__()
        self.channels = channels
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm = StyleIN(channels, (1,), max_styles=max_styles)
        self.act = nn.ReLU()
        self.rezero = nn.Parameter(torch.Tensor([0.]))
        self.recover_shape = recover_shape
    def forward(self, x, y,):
        goal_shape = x.size()
        x_1d = x.permute(0, 3, 2, 1).reshape(goal_shape[0], -1, self.channels)
        res = self.act(self.norm(x_1d, y))
        res, _ = self.attn(res, res, res)
        out = (res * self.rezero) + x_1d
        if self.recover_shape:
            out = out.reshape(goal_shape[0], goal_shape[3], goal_shape[2], goal_shape[1]).permute(0, 3, 2, 1)
        return out
        
class Block(nn.Module):
    def __init__(self, input_channels, output_channels, strides, max_styles):
        super().__init__()
        self.conv = nn.Conv1d(input_channels,
                              output_channels,
                              kernel_size=2*strides,
                              stride=strides,
                              padding=0)
        self.norm = StyleIN(input_channels, (1,), max_styles=max_styles)
        self.act = nn.ReLU()
        self.rezero = nn.Parameter(torch.Tensor([0.]))
        self.skip = nn.Identity() if input_channels == output_channels else nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=2, padding=0)
    def forward(self, x, y):
        res = self.conv(self.act(self.norm(x, y)))
        out = (res * self.rezero) + self.skip(self.pool(x))
        return out


class Module(nn.Module):
    def __init__(self, num_layers, input_channels, embedding_dim, num_poses):
        super(Module, self).__init__()
        out_channels = [input_channels] + [min(64 * 2**i, embedding_dim) for i in range(num_layers)]

        channels = list(zip(out_channels[:-1], out_channels[1:-1] + [embedding_dim]))

        self.patches = SequentialWithArgs(*[
            PatchBlock(channels[i][0], channels[i][1], strides=2, max_styles=num_poses)
            for i in range(num_layers)])
        
        self.mix = AutoConvertAttnBlock(embedding_dim, embedding_dim//128, max_styles=num_poses, recover_shape=False)
        self.final = Block(embedding_dim, embedding_dim, strides=2, max_styles=num_poses)
        self.finish = nn.Sequential(nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
        self.embdim = embedding_dim
    
    def forward(self, x, pose):
        reduced = self.patches(x, pose)
        tensor = self.mix(reduced, pose)
        #tensor = tensor.permute(0, 3, 2, 1).reshape(tensor.size(0), -1, self.embdim)
        output = self.finish(tensor)
        return output
        
        
