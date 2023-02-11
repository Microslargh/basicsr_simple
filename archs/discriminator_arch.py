import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from basicsr.utils.registry import ARCH_REGISTRY
# from basicsr.archs.swin_transformer_unet_skip_expand_decoder_sysfujian import SUNet
from basicsr.archs.swin_transformer_unet_skip_expand_decoder_sys_guanwang import SwinTransformerSys
from basicsr.archs.SUNet_detail import SUNet

@ARCH_REGISTRY.register()
class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 128 x 128 or 256 x 256.

    It is used to train SRGAN, ESRGAN, and VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.Default: 64.
    """

    def __init__(self, num_in_ch, num_feat, input_size=128):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size
        assert self.input_size == 128 or self.input_size == 256, (
            f'input size must be 128 or 256, but received {input_size}')

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        if self.input_size == 256:
            self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
            self.bn5_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
            self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn5_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

        if self.input_size == 256:
            feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
            feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: / 64

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


@ARCH_REGISTRY.register(suffix='basicsr')
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


@ARCH_REGISTRY.register(suffix='basicsr')
class SwinUnet_liu(nn.Module):
    '''
    https://github.com/HuCaoFighting/Swin-Unet/blob/main/configs/swin_tiny_patch4_window7_224_lite.yaml

    net = Swinunet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    config --> options/*.yaml

    DROP_PATH_RATE: 0.2
    FINAL_UPSAMPLE: "expand_first"
    EMBED_DIM: 96
    DEPTHS: [ 2, 2, 2, 2 ]
    DECODER_DEPTHS: [ 2, 2, 2, 1]
    NUM_HEADS: [ 3, 6, 12, 24 ]
    WINDOW_SIZE: 7

    img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    '''
    '''
    def __init__(self, img_size=256,
                num_classes=64,
                zero_head=False,
                vis=False,
                DROP_PATH_RATE=0.2,
                EMBED_DIM=96,
                DEPTHS=[ 2, 2, 2, 2 ],
                DECODER_DEPTHS=[ 2, 2, 2, 1],
                NUM_HEADS=[ 3, 6, 12, 24 ],
                WINDOW_SIZE=7,
                gt_size=64
                ):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.DROP_PATH_RATE = DROP_PATH_RATE
        self.EMBED_DIM = EMBED_DIM
        self.DEPTHS = DEPTHS
        self.DECODER_DEPTHS = DECODER_DEPTHS
        self.NUM_HEADS = NUM_HEADS
        self.WINDOW_SIZE = WINDOW_SIZE
        self.gt_size = gt_size


        self.swin_unet = SwinTransformerSys(img_size=self.gt_size,
                                #patch_size=self.PATCH_SIZE,
                                #in_chans=IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=self.EMBED_DIM,
                                depths=self.DEPTHS,
                                num_heads=self.NUM_HEADS,
                                window_size=self.WINDOW_SIZE,
                                #mlp_ratio=MLP_RATIO,
                                #qkv_bias=QKV_BIAS,
                                #qk_scale=QK_SCALE,
                                #drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=self.DROP_PATH_RATE,
                                #ape=APE,
                                #patch_norm=PATCH_NORM,
                                #use_checkpoint=config.TRAIN.USE_CHECKPOINT
                                )
    '''
    def __init__(self, img_size=256, num_classes=64, zero_head=False, vis=False):
        super(SwinUnet_liu, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        # self.config = config
        # depths=[2262]#
        # numclass=1

        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                patch_size=4,
                                in_chans=3,
                                num_classes=self.num_classes,
                                embed_dim=96,
                                depths=[2,2,2,2],
                                depths_decoder=[2, 2, 2, 1],
                                num_heads=[3,6,12,24],
                                window_size=8,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


@ARCH_REGISTRY.register()
class SwinUnet(nn.Module):
    '''
    https://github.com/HuCaoFighting/Swin-Unet/blob/main/configs/swin_tiny_patch4_window7_224_lite.yaml

    net = Swinunet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    config --> options/*.yaml

    DROP_PATH_RATE: 0.2
    FINAL_UPSAMPLE: "expand_first"
    EMBED_DIM: 96
    DEPTHS: [ 2, 2, 2, 2 ]
    DECODER_DEPTHS: [ 2, 2, 2, 1]
    NUM_HEADS: [ 3, 6, 12, 24 ]
    WINDOW_SIZE: 7

    img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):

    parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')

    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')

    '''
    def __init__(self, img_size=256,patch_size=4, in_chans=3,out_chans=1,
                zero_head=False,
                APE=False,patch_norm=True,
                DROP_PATH_RATE=0.1,DROP_RATE=0.,
                embed_dim=64,attn_drop_rate=0.,
                DEPTHS=[ 8, 8, 8, 8 ],final_upsample="Dual up-sample",
                NUM_HEADS=[ 8, 8, 8, 8 ],norm_layer=nn.LayerNorm,
                WINDOW_SIZE=8,mlp_ratio=4., qkv_bias=True, qk_scale=2,use_checkpoint=False
                ):
        super(SwinUnet, self).__init__()
        self.out_chans = out_chans
        self.zero_head = zero_head
        self.PATCH_SIZE = patch_size
        # self.in_chans = in_chans
        self.DROP_RATE = DROP_RATE
        self.DROP_PATH_RATE = DROP_PATH_RATE
        self.EMBED_DIM = embed_dim
        self.DEPTHS = DEPTHS
        self.NUM_HEADS = NUM_HEADS
        self.WINDOW_SIZE = WINDOW_SIZE
        # self.img_size = img_size
        self.MLP_RATIO = mlp_ratio
        self.QKV_BIAS = qkv_bias
        self.QK_SCALE = qk_scale
        self.ATTN_DROP_RATE=attn_drop_rate
        self.norm_layer=norm_layer
        self.APE=APE
        self.patch_norm=patch_norm
        self.use_checkpoint=use_checkpoint
        self.final=final_upsample


        self.swin_unet = SUNet(img_size=img_size,
                                patch_size=self.PATCH_SIZE,
                                in_chans=in_chans,out_chans=self.out_chans,
                                embed_dim=self.EMBED_DIM,
                                depths=self.DEPTHS,
                                num_heads=self.NUM_HEADS,
                                window_size=self.WINDOW_SIZE,
                                mlp_ratio=self.MLP_RATIO,
                                qkv_bias=self.QKV_BIAS,
                                qk_scale=self.QK_SCALE,norm_layer=self.norm_layer,
                                drop_rate=self.DROP_RATE,attn_drop_rate=self.ATTN_DROP_RATE,
                                drop_path_rate=self.DROP_PATH_RATE,
                                ape=self.APE,
                                patch_norm=self.patch_norm,final_upsample=self.final,
                                use_checkpoint=self.use_checkpoint
                                )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
