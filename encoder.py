import torch
import torch.nn as nn
import kornia
#from torch.utils.checkpoint import checkpoint
from modules import ConvReduceModule


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


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x

class ImageEmbedder(nn.Module):
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=True,
            ucg_rate=0.
    ):
        super().__init__()
        #from clip import load as load_clip
        #self.model, _ = load_clip(name=model, device=device, jit=jit)
        self.model = model

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # re-normalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x, no_dropout=False):
        # x is assumed to be in range [-1,1]
        out = self.model.encode_image(self.preprocess(x))
        out = out.to(x.dtype)
        if self.ucg_rate > 0. and not no_dropout:
            out = torch.bernoulli((1. - self.ucg_rate) * torch.ones(out.shape[0], device=out.device))[:, None] * out
        return out


class LightingEncoder(AbstractEncoder):
    def __init__(self, num_layers, input_channels, embedding_dim, num_poses, light_pos_maps=None, device="cuda"):
        super().__init__()
        # TODO light position -> 2d image level로 light map을 전사하기
        if light_pos_maps is None:
            self.register_buffer('light_condition_table', nn.Parameter(torch.Tensor(), requires_grad=False))
        else:
            self.register_buffer('light_condition_table', nn.Parameter(light_pos_maps, requires_grad=False))

        self.light_condition_table.to(device)
        self.light_module = ConvReduceModule(num_layers, input_channels, embedding_dim, num_poses).to(device)
        #self.light_condition_encoder = ImageEmbedder(device, light_module)
        # print(f"{self.light_module.__class__.__name__} has {count_params(self.light_condition_encoder) * 1.e-6:.2f} M parameters, ")

    def lookup_light_condition(self, label):
        return self.light_condition_table[label.to(self.light_condition_table.device).long()-1]

    def encode(self, light_label, angle_label):
        light_map = self.lookup_light_condition(light_label)
        light_cond = self.light_module(light_map, angle_label)
        return light_cond

    def forward(self, light_label, angle_label):
        return self.encode(light_label, angle_label)