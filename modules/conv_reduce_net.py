import torch
import torch.nn as nn
import itertools

class Module(nn.Module):
    def __init__(self, num_layers, input_channels, embedding_dim, num_poses,):
        super(Module, self).__init__()
        out_channels = [input_channels] + [min(64 * 2**i, embedding_dim) for i in range(num_layers)]
        # print(out_channels)
        channels = list(zip(out_channels[:-1], out_channels[1:-1] + [embedding_dim]))
        layers = list(itertools.chain.from_iterable([(
            nn.Conv2d(channels[i][0], channels[i][1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels[i][1]),
            nn.ReLU()) for i in range(num_layers)])) + [
            nn.AdaptiveAvgPool2d((1, 1)),  # 출력 크기: (B, C, 1, 1)
            nn.Flatten()]
        
        self.image_processing = nn.Sequential(*layers)
        self.label_embedding = nn.Embedding(num_poses, embedding_dim * 2)
        self.act_proj = nn.ReLU()
        self.lin_proj = nn.Linear(embedding_dim, embedding_dim)

    
    def forward(self, x, pose):
        img_factor = self.image_processing(x)
        pose_amp, pose_shift = self.label_embedding(pose).chunk(2, dim=1)
        total_factor = pose_amp * (img_factor + pose_shift)
        output = self.lin_proj(self.act_proj(total_factor))
        return output
        
        
