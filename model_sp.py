"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Sp_attention(nn.Module):
    def __init__(self, img_size, dim):
        super().__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(dim, dim//4, kernel_size=3, padding=1)
        self.norm1 = LayerNorm(dim//4, eps=1e-6, data_format="channels_first")
        self.conv2 = nn.Conv2d(dim//4, 1, kernel_size=3, padding=1)
        self.norm2 = LayerNorm(1, eps=1e-6, data_format="channels_first")
        self.Fl = nn.Flatten(1)
        self.fc1 = nn.Linear(img_size ** 2, img_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(img_size, img_size ** 2)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = self.norm2(self.conv2(x))
        x = self.Fl(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = x.reshape(x.shape[0], 1, self.img_size, self.img_size)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0.4, layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            if i == 3:
                cur_ = cur
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)
        self.head = nn.Linear(dims[-1] + dims[-2] + dims[-3], num_classes)
        self.head1 = nn.Linear(dims[-1] + dims[-2] + dims[-3], 512)
        self.head2 = nn.Linear(512, num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        # Sp注意力
        img_size = [56, 28, 14, 7]
        self.sp_att = nn.ModuleList([
            Sp_attention(img_size[0], dims[0]),
            Sp_attention(img_size[1], dims[1]),
            Sp_attention(img_size[2], dims[2]),
            Sp_attention(img_size[3], dims[3])
        ])
        self.norm_att = nn.ModuleList([
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
            LayerNorm(dims[3], eps=1e-6, data_format="channels_first")
        ])

        # FPN
        downsample_layer = nn.Sequential(LayerNorm(dims[3], eps=1e-6, data_format="channels_first"),
                                         nn.Conv2d(dims[3], dims[2 + 1], kernel_size=3, stride=1, padding=1))
        self.downsample_layers.append(downsample_layer)
        stage = nn.Sequential(
            *[Block(dim=dims[3], drop_rate=dp_rates[cur_ + j], layer_scale_init_value=layer_scale_init_value)
              for j in range(depths[3])]
        )
        self.stages.append(stage)

        self.upsample_layers = nn.ModuleList([
            nn.Sequential(LayerNorm(dims[3], eps=1e-6, data_format="channels_first"),
                          nn.Conv2d(in_channels=dims[3], out_channels=dims[3], kernel_size=3, padding=1)),
            nn.Sequential(LayerNorm(dims[3], eps=1e-6, data_format="channels_first"),
                          nn.ConvTranspose2d(in_channels=dims[3], out_channels=dims[2], kernel_size=2, stride=2)),
            nn.Sequential(LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
                          nn.ConvTranspose2d(in_channels=dims[2], out_channels=dims[1], kernel_size=2, stride=2)),
        ])

        self.upsatges = nn.ModuleList([
            Block(dim=dims[3], drop_rate=dp_rates[-1], layer_scale_init_value=layer_scale_init_value),
            Block(dim=dims[2], drop_rate=dp_rates[-1], layer_scale_init_value=layer_scale_init_value),
            Block(dim=dims[1], drop_rate=dp_rates[-1], layer_scale_init_value=layer_scale_init_value),
        ])

        self.up_sp_att = nn.ModuleList([
            Sp_attention(img_size[3], dims[3]),
            Sp_attention(img_size[2], dims[2]),
            Sp_attention(img_size[1], dims[1])
        ])
        self.up_cat_att = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Conv2d(2, 1, kernel_size=3, padding=1)
        ])

        self.norm_fpn = nn.LayerNorm(dims[3] + dims[2] + dims[1], eps=1e-6)  # final norm layer


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        a = [0, 0, 0, 0]
        t = [0, 0, 0]
        att = [0, 0, 0, 0]
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            a[i] = x
            # Sp_att
            if i > 0:
                att[i] = self.sp_att[i](x)
                x = self.norm_att[i](x * att[i])

        # FPN
        x = self.downsample_layers[4](x)
        x = self.stages[4](x)

        for i in range(3):
            x = self.upsample_layers[i](x)
            x = x + a[3-i]
            att_up = self.up_sp_att[i](x)
            att_up = torch.cat((att_up, att[3-i]), dim=1)
            att_up = self.up_cat_att[i](att_up)
            x = self.norm_att[3-i]((x + a[3-i]) * att_up)
            # x = self.norm_att[3-i]((x + a[3-i]) * att[3-i])
            x = self.upsatges[i](x)
            t[i] = x.mean([-2, -1])

        # return [t[2], t[1], t[0]]

        return self.norm_fpn(torch.cat((t[0], t[1], t[2]), dim=-1))

        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.forward_features(x)
        # x = self.head(x)
        # x = self.head2(self.head1(x))
        # x = x.permute(1, 0, 2, 3, 4)
        x = [self.forward_features(t) for t in x]
        # x = [torch.stack([t[i] for t in x], dim=0) for i in range(len(x[0]))]
        x = torch.stack(x, dim=0)
        # x = [torch.stack(x[0], dim=0), torch.stack(x[1], dim=0), torch.stack(x[2], dim=0)]
        # x = x.permute(1, 0, 2)
        return x


def convnext_tiny(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def convnext_small(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     # depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def convnext_base(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     # depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes)
    return model


def convnext_large(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes)
    return model


def convnext_xlarge(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes)
    return model


class Attention(nn.Module):
    def __init__(self, num_classes):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(1024, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, num_classes),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part2(x)  # KxM

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        # return Y_prob, Y_hat, A
        return Y_prob

class CA_AbMIL(nn.Module):
    def __init__(self, num_classes):
        super(CA_AbMIL, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(768, self.M),
            # nn.Linear(200 * 14 * 14, self.M),
            nn.GELU(),
        )
        self.feature_extractor_part1 = nn.ParameterList([
            nn.Sequential(nn.Linear(768, 384), nn.GELU(), nn.Linear(384, 768), nn.GELU()),
            nn.Sequential(nn.Linear(1344, 672), nn.GELU(), nn.Linear(672, 1344), nn.GELU()),
            nn.Sequential(nn.Linear(1344, 672), nn.GELU(), nn.Linear(672, 1344), nn.GELU()),
        ])

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(768, self.M),
            nn.GELU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, num_classes),
        )

    def forward(self, x):
        # x = x.squeeze(0)


        query = self.feature_extractor_part1[0](x[2])
        keys = self.feature_extractor_part1[1](torch.cat((x[0], x[1], x[2]), dim=2))
        values = self.feature_extractor_part1[2](torch.cat((x[0], x[1], x[2]), dim=2))

        # 计算注意力权重
        attention_weights = torch.matmul(query.transpose(1, 2), keys)  # 形状为 (1, 4, 15)
        attention_weights = torch.softmax(attention_weights, dim=-1)  # 归一化

        # 加权求和
        weighted_sum = torch.matmul(attention_weights, values.transpose(1, 2)).transpose(1, 2).squeeze(0)  # 形状为 (1, 4, 10)

        H = self.feature_extractor_part2(weighted_sum)  # KxM

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        # return Y_prob, Y_hat, A
        return Y_prob