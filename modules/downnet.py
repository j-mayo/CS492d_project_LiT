import torch
import torch.nn as nn
import itertools

from typing import Any, Dict, List, Optional, Tuple, Union
from termcolor import cprint
from diffusers.models.resnet import ResnetBlockCondNorm2D, ResnetBlock2D, Downsample2D
from diffusers.models.attention_processor import Attention, AttnAddedKVProcessor, AttnAddedKVProcessor2_0


# copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py#L1027
# modified return type, and whether to downsample
class AttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.downsample_type = downsample_type

        if attention_head_dim is None:
            logger.warning(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
            )
            attention_head_dim = out_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if downsample_type == "conv":
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        elif downsample_type == "resnet":
            self.downsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        down=True,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        upsample_size: Optional[int] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_hidden_states = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, **cross_attention_kwargs)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                if self.downsample_type == "resnet":
                    hidden_states = downsampler(hidden_states, temb=temb)
                else:
                    hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        if return_hidden_states:
            return hidden_states
        else:
            return hidden_states, output_states


class SequentialWithArgs(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input


class Module(nn.Module):
    def __init__(self, num_layers, input_channels, embedding_dim, num_poses, pose_channels):
        super(Module, self).__init__()
        self.proj = nn.Conv2d(input_channels, 16, kernel_size=3, padding=(1, 1))
        ch = [16]
        for i in range(num_layers):
            ch.append(min(ch[-1]*2, embedding_dim))
        channels = list(zip(ch[:-1], ch[1:]))
        self.blocks = SequentialWithArgs(*[#AttnDownBlock2D(in_channels=cin,
                                           #                out_channels=cout,
                                           #                temb_channels=pose_channels,
                                           #                num_layers=1,
                                           #                attention_head_dim=max(2, cout // 128))
                                           ResnetBlock2D(in_channels=cin,
                                                         out_channels=cout,
                                                         temb_channels=pose_channels,
                                                         eps=1e-6,
                                                         groups=min(32, cin//4),
                                                         time_embedding_norm="default",
                                                         non_linearity="swish",
                                                         pre_norm=True,
                                                         down=True,)
                                           for cin, cout in channels[:-1]] + [
                                           AttnDownBlock2D(in_channels=ch[-2],
                                                           out_channels=embedding_dim,
                                                           temb_channels=pose_channels,
                                                           num_layers=2,
                                                           attention_head_dim=8,
                                                           downsample_type=None,)])
        #self.last_block = ResnetBlock2D(in_channels=ch[-1],
        #                                out_channels=embedding_dim,
        #                                temb_channels=pose_channels,
        #                                eps=1e-6,
        #                                groups=32,
        #                                time_embedding_norm="default",
        #                                non_linearity="swish",
        #                                pre_norm=True,
        #                                down=True,)
        self.last_pool = nn.AdaptiveAvgPool1d(64)
        self.finish = nn.Sequential(nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
        self.embdim = embedding_dim
    
    def forward(self, x, v_pose):
        tensor = self.proj(x)
        #for blk in self.blocks:
        tensor = self.blocks(tensor, v_pose)
        tensor = tensor.view(tensor.size(0), tensor.size(1), -1)
        tensor = self.last_pool(tensor)
        tensor = tensor.permute(0, 2, 1)
        output = self.finish(tensor)
        return output
        
        
