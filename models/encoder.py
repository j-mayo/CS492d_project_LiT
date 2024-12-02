import os
import json
import torch
import torch.nn as nn
import kornia

from inspect import isfunction
from modules import AttnDownNet
from glob import glob
from termcolor import cprint
opj = os.path.join

def exists(val): return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params

def autocast(f):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(enabled=True,
                                     dtype=torch.get_autocast_gpu_dtype(),
                                     cache_enabled=torch.is_autocast_cache_enabled()):
            return f(*args, **kwargs)

    return do_autocast


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config_path, subfolder=None, **kwargs):
        # for simple config initialization, use dict
        fname = "config.json"
        if exists(subfolder):
            config_path = opj(config_path, subfolder)
            fname = "_".join([subfolder, fname])
        config_file = opj(config_path, fname)
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
        except Exception as e:
            raise e

        initialized = cls(**config_dict, **kwargs)
        initialized.config = config_dict
        return initialized


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class LightingEncoder(AbstractEncoder):
    def __init__(self, num_layers, input_channels, embedding_dim, num_poses, light_pos_maps=None, device="cuda"):
        super().__init__()
        # light position -> 2d image level로 light map을 전사하기
        self.register_buffer('light_condition_table',
                             nn.Parameter(default(light_pos_maps, torch.Tensor()), requires_grad=False))
        self.light_condition_table = self.light_condition_table.contiguous().to(device)

        self.pose_emb = nn.Embedding(num_poses, embedding_dim//4)

        self.lcond_pooler = nn.AvgPool2d(4, 2)
        self.light_module = AttnDownNet(num_layers=num_layers,
                                        input_channels=input_channels,
                                        embedding_dim=embedding_dim // 2,
                                        num_poses=num_poses,
                                        pose_channels=embedding_dim // 4).to(device)

    def set_condition_table(self, tensor):
        self.register_buffer('light_condition_table',
                             nn.Parameter(tensor.contiguous(), requires_grad=False))

    def lookup_light_condition(self, label, base_label=None):
        lcond = self.light_condition_table[label.to(self.light_condition_table.device).long()-1]
        return self.lcond_pooler(lcond)

    def encode(self, light_label, angle_label):
        angle_emb = self.pose_emb(angle_label)
        light_map = self.lookup_light_condition(light_label)
        light_cond = self.light_module(light_map, angle_emb)
        return light_cond

    def forward(self, src_light_label, tgt_light_label, angle_label):
        src_emb = self.encode(src_light_label, angle_label)
        tgt_emb = self.encode(tgt_light_label, angle_label)
        return torch.cat([src_emb, tgt_emb], dim=2)

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        missing_keys = []
        unexpected_keys = []

        if 'light_condition_table' in state_dict.keys():
            self.set_condition_table(state_dict['light_condition_table'])
            state_dict.pop('light_condition_table')
        else:
            if strict:
                missing_keys.append('light_condition_table')

        result = super().load_state_dict(state_dict, strict=False)
        missing_keys.extend(result.missing_keys)
        unexpected_keys.extend(result.unexpected_keys)

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n"
                f"\tMissing keys: {missing_keys}\n"
                f"\tUnexpected keys: {unexpected_keys}"
            )

        return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)


class LatentEncoder(AbstractEncoder):
    def __init__(self, num_layers, input_channels, embedding_dim, num_poses, device="cuda"):
        super().__init__()
        self.pose_emb = nn.Embedding(num_poses, embedding_dim//4)
        self.module = AttnDownNet(num_layers=num_layers+1,
                                  input_channels=input_channels,
                                  embedding_dim=embedding_dim,
                                  num_poses=num_poses,
                                  pose_channels=embedding_dim//4).to(device)
    def forward(self, lat, angle_label):
        angle_emb = self.pose_emb(angle_label)
        cond = self.module(lat, angle_emb)
        return cond

class CompoundEncoder(AbstractEncoder):
    def __init__(self,
                 lighting_layers,
                 latent_layers,
                 light_channels,
                 latent_channels,
                 embedding_dim,
                 num_poses,
                 light_pos_maps=None,
                 device="cuda"):
        super().__init__()
        self.lighting_encoder = LightingEncoder(lighting_layers,
                                                light_channels,
                                                embedding_dim,
                                                num_poses,
                                                light_pos_maps,
                                                device=device)
        self.latent_encoder = LatentEncoder(latent_layers,
                                            latent_channels,
                                            embedding_dim,
                                            num_poses,
                                            device=device)

        self.act = nn.ReLU()
        self.linear = nn.Linear(2 * embedding_dim, embedding_dim)
        light_p = count_params(self.lighting_encoder)
        latent_p = count_params(self.latent_encoder)

    def forward(self, image, src_light, tgt_light, angle):
        light_cond = self.lighting_encoder(src_light, tgt_light, angle)
        lat_cond = self.latent_encoder(image, angle)
        return self.linear(self.act(torch.cat([light_cond, lat_cond], dim=2)))

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        missing_keys = []
        unexpected_keys = []

        if 'lighting_encoder.light_condition_table' in state_dict.keys():
            self.lighting_encoder.set_condition_table(
                    state_dict['lighting_encoder.light_condition_table'])
            state_dict.pop('lighting_encoder.light_condition_table')
        else:
            if strict:
                missing_keys.append('lighting_encoder.light_condition_table')

        result = super().load_state_dict(state_dict, strict=False)
        missing_keys.extend(result.missing_keys)
        unexpected_keys.extend(result.unexpected_keys)

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n"
                f"\tMissing keys: {missing_keys}\n"
                f"\tUnexpected keys: {unexpected_keys}"
            )

        return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)

