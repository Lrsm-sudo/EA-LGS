# """
# Channel and Spatial CSNet Network (CS-Net).
# """
# from __future__ import division
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # from mmcv.ops.carafe import carafe
# import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#
#
#
# # 添加替代实现
# try:
#     from mmcv.ops.carafe import carafe
#     print("carafe loaded successfully")
# except ImportError as e:
#     print(f"carafe import failed: {e}, using bilinear upsample")
#     import torch.nn.functional as F
#     def carafe(features, scale_factor=2, kernel_size=3, group=1):
#         return F.interpolate(features, scale_factor=scale_factor, mode='bilinear', align_corners=True)
#
#
#
#
# class conv(nn.Module):
#     def __init__(self, in_c, out_c, dp=0):
#         super(conv, self).__init__()
#         self.in_c = in_c
#         self.out_c = out_c
#         self.conv = nn.Sequential(
#             nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.Dropout2d(dp),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.Dropout2d(dp),
#             nn.LeakyReLU(0.1, inplace=True))
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class feature_fuse(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(feature_fuse, self).__init__()
#         self.conv11 = nn.Conv2d(
#             in_c, out_c, kernel_size=1, padding=0, bias=False)
#         self.conv33 = nn.Conv2d(
#             in_c, out_c, kernel_size=3, padding=1, bias=False)
#         self.conv33_di = nn.Conv2d(
#             in_c, out_c, kernel_size=3, padding=2, bias=False, dilation=2)
#         self.norm = nn.BatchNorm2d(out_c)
#
#     def forward(self, x):
#         x1 = self.conv11(x)
#         x2 = self.conv33(x)
#         x3 = self.conv33_di(x)
#         out = self.norm(x1 + x2 + x3)
#         return out
#
#
# class up(nn.Module):
#     def __init__(self, in_c, out_c, dp=0):
#         super(up, self).__init__()
#         self.up = nn.Sequential(
#             nn.ConvTranspose2d(in_c, out_c, kernel_size=2,
#                                padding=0, stride=2, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.LeakyReLU(0.1, inplace=False))
#
#     def forward(self, x):
#         x = self.up(x)
#         return x
#
#
# class down(nn.Module):
#     def __init__(self, in_c, out_c, dp=0):
#         super(down, self).__init__()
#         self.down = nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size=2,
#                       padding=0, stride=2, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.LeakyReLU(0.1, inplace=True))
#
#     def forward(self, x):
#         x = self.down(x)
#         return x
#
#
#
# class block(nn.Module):
#     def __init__(self, in_c, out_c, dp=0, is_up=False, is_down=False, fuse=False):
#         super(block, self).__init__()
#         self.in_c = in_c
#         self.out_c = out_c
#         if fuse == True:
#             self.fuse = feature_fuse(in_c, out_c)
#         else:
#             self.fuse = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)
#
#         self.is_up = is_up
#         self.is_down = is_down
#         self.conv = conv(out_c, out_c, dp=dp)
#         if self.is_up == True:
#             self.up = up(out_c, out_c // 2)
#         if self.is_down == True:
#             self.down = down(out_c, out_c * 2)
#
#     def forward(self, x):
#         if self.in_c != self.out_c:
#             x = self.fuse(x)
#         x = self.conv(x)
#         if self.is_up == False and self.is_down == False:
#             return x
#         elif self.is_up == True and self.is_down == False:
#             x_up = self.up(x)
#             return x, x_up
#         elif self.is_up == False and self.is_down == True:
#             x_down = self.down(x)
#             return x, x_down
#         else:
#             x_up = self.up(x)
#             x_down = self.down(x)
#             return x, x_up, x_down
#
#
#
# # def hamming2D(M, N):
# #     hamming_x = np.hamming(M)
# #     hamming_y = np.hamming(N)
# #     hamming_2d = np.outer(hamming_x, hamming_y)
# #     return hamming_2d
# #
# #
# # class AHPF(nn.Module):
# #     def __init__(self, channels, kernel_size=3, encoder_dilation=1, up_group=1, scale_factor=1):
# #         super().__init__()
# #         self.kernel_size = kernel_size
# #         self.encoder_dilation = encoder_dilation
# #         self.up_group = up_group
# #         self.scale_factor = scale_factor
# #         self.compressed_channels = channels
# #
# #         # 用于高通滤波掩码的生成器（内容编码器）
# #         self.content_encoder2 = nn.Conv2d(
# #             in_channels=self.compressed_channels,
# #             out_channels=kernel_size ** 2 * up_group * scale_factor * scale_factor,
# #             kernel_size=3,
# #             padding=(kernel_size - 1) * encoder_dilation // 2,
# #             dilation=encoder_dilation,
# #             groups=1
# #         )
# #
# #         # 注册汉明窗（hamming_window）用于边缘细节增强
# #         hamming_highpass = torch.FloatTensor(hamming2D(kernel_size, kernel_size))[None, None,]
# #         self.register_buffer('hamming_highpass', hamming_highpass.to(device))  # 移动到 GPU
# #
# #     def forward(self, hr_feat):
# #         # 将输入特征移到 GPU
# #         hr_feat = hr_feat.to(device)
# #
# #         # 生成高通滤波掩码
# #         mask_hr_hr_feat = self.content_encoder2(hr_feat)
# #
# #         # 使用 kernel_normalizer 正规化掩码
# #         mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.kernel_size, hamming=self.hamming_highpass)
# #
# #         # 自适应高通滤波器的操作
# #         hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr_init, self.kernel_size, self.up_group, 1)
# #
# #         return hr_feat_hf
# #
# #     def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
# #         if scale_factor is not None:
# #             mask = F.pixel_shuffle(mask, scale_factor)
# #         n, mask_c, h, w = mask.size()
# #         mask_channel = int(mask_c / float(kernel ** 2))
# #
# #         # 进行 softmax 归一化处理
# #         mask = mask.view(n, mask_channel, -1, h, w)
# #         mask = F.softmax(mask, dim=2, dtype=mask.dtype)
# #         mask = mask.view(n, mask_channel, kernel, kernel, h, w)
# #         mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
# #         mask = mask * hamming
# #         mask /= mask.sum(dim=(-1, -2), keepdims=True)
# #         mask = mask.view(n, mask_channel, h, w, -1)
# #         mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
# #
# #         return mask
#
# def normal_init(module, mean=0, std=1, bias=0):
#     if hasattr(module, 'weight') and module.weight is not None:
#         nn.init.normal_(module.weight, mean, std)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias, bias)
#
#
# def hamming2D(M, N):
#     hamming_x = np.hamming(M)  # 生成长度为 M 的汉明窗
#     hamming_y = np.hamming(N)  # 生成长度为 N 的汉明窗
#     hamming_2d = np.outer(hamming_x, hamming_y)  # 使用外积生成 2D 汉明窗
#     return hamming_2d  # 返回 2D 汉明窗
#
#
# class AHPF(nn.Module):
#     def __init__(self, channels, kernel_size=3, encoder_dilation=1, up_group=1, scale_factor=1, use_high_pass=True):
#         super().__init__()
#         self.kernel_size = kernel_size  # 核大小，用于定义高通滤波的范围
#         self.encoder_dilation = encoder_dilation  # 膨胀率，用于控制特征扩展范围
#         self.up_group = up_group  # 分组数，用于调整高通滤波的分组卷积
#         self.scale_factor = scale_factor  # 缩放因子，用于调整特征尺寸
#         self.compressed_channels = channels  # 输入通道数
#         self.use_high_pass = use_high_pass
#
#         # 内容编码器，用于生成高通滤波掩码
#         self.content_encoder2 = nn.Conv2d(
#             in_channels=self.compressed_channels,  # 输入通道数
#             out_channels=kernel_size ** 2 * up_group * scale_factor * scale_factor,  # 输出通道数，决定了高通滤波掩码的复杂度
#             kernel_size=3,  # 卷积核大小
#             padding=(kernel_size - 1) * encoder_dilation // 2,  # 填充大小，确保输出特征图尺寸一致
#             dilation=encoder_dilation,  # 膨胀率，用于扩展卷积感受野
#             groups=1  # 分组数，标准卷积不分组
#         )
#
#         # 注册汉明窗（hamming_window），用于增强边缘细节
#         hamming_highpass = torch.FloatTensor(hamming2D(kernel_size, kernel_size))[None, None,]  # 生成 2D 汉明窗并扩展维度
#         self.register_buffer('hamming_highpass', hamming_highpass.to(device))  # 将汉明窗注册为缓冲区，并移动到 GPU
#         self.init_weights()
#
#     def init_weights(self):
#         if self.use_high_pass:
#             normal_init(self.content_encoder2, std=0.001)
#
#     def forward(self, hr_feat):
#         # 将输入特征移到 GPU
#         hr_feat = hr_feat.to(device)
#
#         # 生成高通滤波掩码
#         mask_hr_hr_feat = self.content_encoder2(hr_feat)
#
#         # 使用 kernel_normalizer 对掩码进行正规化
#         mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.kernel_size, hamming=self.hamming_highpass)
#
#         # 自适应高通滤波器操作：减去基于掩码的下采样特征
#         hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr_init, self.kernel_size, self.up_group, 1)
#
#         return hr_feat_hf + hr_feat  # 返回高通滤波后的特征
#
#     def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
#         # if scale_factor is not None:
#         #     mask = F.pixel_shuffle(mask, scale_factor)  # 使用像素重排进行上采样
#         n, mask_c, h, w = mask.size()  # 获取掩码的维度信息
#         mask_channel = int(mask_c / float(kernel ** 2))  # 计算每个通道对应的掩码数量
#
#         # 将掩码按照 softmax 进行归一化处理
#         mask = mask.view(n, mask_channel, -1, h, w)  # 调整掩码形状
#         mask = F.softmax(mask, dim=2, dtype=mask.dtype)  # 对掩码权重进行 softmax 归一化
#         mask = mask.view(n, mask_channel, kernel, kernel, h, w)  # 恢复形状
#         mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)  # 调整掩码维度顺序以匹配操作
#         mask = mask * hamming  # 按汉明窗进行加权
#         mask /= mask.sum(dim=(-1, -2), keepdims=True)  # 再次归一化以确保和为 1
#         mask = mask.view(n, mask_channel, h, w, -1)  # 恢复形状
#         mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()  # 转换为最终输出形状
#
#         return mask  # 返回正规化后的掩码
#
#
# class ResConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResConv, self).__init__()
#         self.conv = nn.Sequential(
#             FilterResponseNormNd(4, in_channels),
#             Mish(),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             FilterResponseNormNd(4, out_channels),
#             Mish(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             FilterResponseNormNd(4, out_channels),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1),
#         )
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.relu = Mish()
#     def forward(self, x):
#         out = self.conv(x)
#         residual = self.conv1x1(x)
#         out = out + residual
#         out = self.relu(out)
#         return out
#
#
# class resblock(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation):
#         super(resblock, self).__init__()
#
#         # 修改 resblock 中的 local_conv
#         self.local_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),  # 普通卷积
#             FilterResponseNormNd(4, in_channels//2),
#             Mish(),
#             nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=dilation, dilation=dilation),  # 空洞卷积
#             FilterResponseNormNd(4, in_channels//2),
#             Mish(),
#             nn.Conv2d(in_channels//2, out_channels, kernel_size=1)
#         )
#
#         # 1x1卷积，用于调整输入通道数
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.relu = Mish()
#
#     def forward(self, x):
#
#         local_features = self.local_conv(x)
#         residual = self.conv1(x)
#         out = local_features + residual
#         out = self.relu(out)
#         return out
#
#
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             Mish(),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return y
#
#
# class MCA(nn.Module):
#     def __init__(self, channel=64, ratio=16):
#         super(MCA, self).__init__()
#         self.ahpf1 = AHPF(channel)
#         self.ahpf2 = AHPF(channel * 2)
#         self.ahpf3 = AHPF(channel * 4)
#         self.ahpf4 = AHPF(channel * 8)
#
#         # 第一层，空洞率为 1
#         self.conv1 = resblock(channel, channel, 1)
#         # 第二层，空洞率为 2
#         self.conv2 = resblock(channel * 2, channel * 2, 2)
#         # 第三层，空洞率为 4
#         self.conv3 = resblock(channel * 4, channel * 4, 4)
#         # 第四层，空洞率为 8
#         self.conv4 = resblock(channel * 8, channel * 8, 5)
#
#         self.se1 = SELayer(channel)
#         self.se2 = SELayer(channel * 2)
#         self.se3 = SELayer(channel * 4)
#         self.se4 = SELayer(channel * 8)
#
#         self.conv1x1 = nn.Conv2d(480, 512, 1)
#
#         self.resconv1 = ResConv(channel, channel)
#         self.resconv2 = ResConv(channel * 2, channel * 2)
#         self.resconv3 = ResConv(channel * 4, channel * 4)
#         self.resconv4 = ResConv(channel * 8, channel * 8)
#
#         self.resconv = ResConv(channel * 16, channel * 16)
#
#     def forward(self, x1, x2, x3, x4):
#
#         x1 = self.ahpf1(x1)
#         x2 = self.ahpf2(x2)
#         x3 = self.ahpf3(x3)
#         x4 = self.ahpf4(x4)
#
#         x1 = self.resconv1(x1)
#         x2 = self.resconv2(x2)
#         x3 = self.resconv3(x3)
#         x4 = self.resconv4(x4)
#
#         x1 = self.conv1(x1)  # 第一层卷积，空洞率 1
#         x2 = self.conv2(x2)  # 第二层卷积，空洞率 2
#         x3 = self.conv3(x3)  # 第三层卷积，空洞率 4
#         x4 = self.conv4(x4)  # 第四层卷积，空洞率 5
#
#         x10 = self.se1(x1)
#         x20 = self.se2(x2)
#         x30 = self.se3(x3)
#         x40 = self.se4(x4)
#
#         weights = [0.4, 0.3, 0.2, 0.1]  # 示例权重
#         x10 = x10 * weights[0]
#         x20 = x20 * weights[1]
#         x30 = x30 * weights[2]
#         x40 = x40 * weights[3]
#         c = torch.cat([x10, x20, x30, x40], dim=1)
#         c = self.conv1x1(c)
#         c = self.resconv(c)
#         return c
#
#
# class FilterResponseNormNd(nn.Module):
#
#     def __init__(self, ndim, num_features, eps=1e-6,
#                  learnable_eps=False):
#         """
#         Input Variables:
#         ----------------
#             ndim: An integer indicating the number of dimensions of the expected input tensor.
#             num_features: An integer indicating the number of input feature dimensions.
#             eps: A scalar constant or learnable variable.
#             learnable_eps: A bool value indicating whether the eps is learnable.
#         """
#         assert ndim in [3, 4, 5], \
#             'FilterResponseNorm only supports 3d, 4d or 5d inputs.'
#         super(FilterResponseNormNd, self).__init__()
#         shape = (1, num_features) + (1,) * (ndim - 2)
#         self.eps = nn.Parameter(torch.ones(*shape) * eps)
#         if not learnable_eps:
#             self.eps.requires_grad_(False)
#         # self.gamma = nn.Parameter(torch.Tensor(*shape))
#         # self.beta = nn.Parameter(torch.Tensor(*shape))
#         # self.tau = nn.Parameter(torch.Tensor(*shape))
#         # self.reset_parameters()
#
#     def forward(self, x):
#         avg_dims = tuple(range(2, x.dim()))  # (2, 3)
#         nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
#         x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
#         return x
#
#     # def reset_parameters(self):
#     #     nn.init.ones_(self.gamma)
#     #     nn.init.zeros_(self.beta)
#     #     nn.init.zeros_(self.tau)
#
#
# class Mish_func(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, i):
#         result = i * torch.tanh(F.softplus(i))
#         ctx.save_for_backward(i)
#         return result
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         i = ctx.saved_variables[0]
#
#         v = 1. + i.exp()
#         h = v.log()
#         grad_gh = 1. / h.cosh().pow_(2)
#
#         # Note that grad_hv * grad_vx = sigmoid(x)
#         # grad_hv = 1./v
#         # grad_vx = i.exp()
#
#         grad_hx = i.sigmoid()
#
#         grad_gx = grad_gh * grad_hx  # grad_hv * grad_vx
#
#         grad_f = torch.tanh(F.softplus(i)) + i * grad_gx
#
#         return grad_output * grad_f
#
#
# class Mish(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         pass
#
#     def forward(self, input_tensor):
#         return Mish_func.apply(input_tensor)
#
#
# def downsample():
#     return nn.MaxPool2d(kernel_size=2, stride=2)
#
#
# def deconv0(in_channels, out_channels):
#
#     return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#
# def deconv1(in_channels, out_channels, upscale_factor=2):
#     # Calculate the input channels required for PixelShuffle
#     # PixelShuffle requires input channels to be in the form of in_channels * upscale_factor^2
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
#         torch.nn.PixelShuffle(upscale_factor)
#     )
#
#
# class ResEncoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResEncoder, self).__init__()
#         self.conv0 = nn.Sequential(
#             FilterResponseNormNd(4, in_channels),
#             Mish(),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             FilterResponseNormNd(4, out_channels),
#             Mish(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             # nn.Dropout(0.2),
#             FilterResponseNormNd(4, out_channels),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1),
#         )
#         self.conv1 = nn.Sequential(
#             FilterResponseNormNd(4, out_channels),
#             Mish(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1),
#             FilterResponseNormNd(4, out_channels),
#             Mish(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             # nn.Dropout(0.2),
#             FilterResponseNormNd(4, out_channels),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1),
#         )
#         self.conv2 = nn.Sequential(
#             FilterResponseNormNd(4, out_channels),
#             Mish(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1),
#             FilterResponseNormNd(4, out_channels),
#             Mish(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.Dropout(0.2),
#             FilterResponseNormNd(4, out_channels),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1),
#         )
#
#
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.relu = Mish()
#
#     def forward(self, x):
#         residual = self.conv1x1(x)
#         out1 = self.conv0(x)
#         out2 = self.conv1(out1)
#         out3 = self.conv2(out2)
#         out = out3 + residual
#         out = self.relu(out)
#         return out
#
#
# class Decoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Decoder, self).__init__()
#         self.conv0 = nn.Sequential(
#             FilterResponseNormNd(4, in_channels),
#             Mish(),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             FilterResponseNormNd(4, out_channels),
#             Mish(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             # nn.Dropout(0.2),
#             FilterResponseNormNd(4, out_channels),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1),
#         )
#
#         self.conv1 = nn.Sequential(
#             FilterResponseNormNd(4, out_channels),
#             Mish(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1),
#             FilterResponseNormNd(4, out_channels),
#             Mish(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             # nn.Dropout(0.2),
#             FilterResponseNormNd(4, out_channels),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1),
#         )
#
#         self.conv2 = nn.Sequential(
#             FilterResponseNormNd(4, out_channels),
#             Mish(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1),
#             FilterResponseNormNd(4, out_channels),
#             Mish(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.Dropout(0.2),
#             FilterResponseNormNd(4, out_channels),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1),
#         )
#
#     def forward(self, x):
#         out = self.conv0(x)
#         out = self.conv1(out)
#         out = self.conv2(out)
#         return out
#
#
# class SpatialAttentionBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(SpatialAttentionBlock, self).__init__()
#         self.query = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 16, kernel_size=(1, 3), padding=(0, 1)),
#             FilterResponseNormNd(4, in_channels // 16),
#             Mish(),
#         )
#         self.key = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 16, kernel_size=(3, 1), padding=(1, 0)),
#             FilterResponseNormNd(4, in_channels // 16),
#             Mish(),
#         )
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         """
#         :param x: input( BxCxHxW )
#         :return: affinity value + x
#         """
#         B, C, H, W = x.size()
#         # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
#         proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
#         proj_key = self.key(x).view(B, -1, W * H)
#         affinity = torch.matmul(proj_query, proj_key)
#         affinity = self.softmax(affinity)
#         proj_value = self.value(x).view(B, -1, H * W)
#         weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
#         weights = weights.view(B, C, H, W)
#         out = self.gamma * weights + x
#         return out
#
#
# class ChannelAttentionBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(ChannelAttentionBlock, self).__init__()
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         """
#         :param x: input( BxCxHxW )
#         :return: affinity value + x
#         """
#         B, C, H, W = x.size()
#         proj_query = x.view(B, C, -1)
#         proj_key = x.view(B, C, -1).permute(0, 2, 1)
#         affinity = torch.matmul(proj_query, proj_key)
#         affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
#         affinity_new = self.softmax(affinity_new)
#         proj_value = x.view(B, C, -1)
#         weights = torch.matmul(affinity_new, proj_value)
#         weights = weights.view(B, C, H, W)
#         out = self.gamma * weights + x
#         return out
#
#
# class AffinityAttention(nn.Module):
#     """ Affinity attention module """
#
#     def __init__(self, in_channels):
#         super(AffinityAttention, self).__init__()
#         self.sab = SpatialAttentionBlock(in_channels)
#         self.cab = ChannelAttentionBlock(in_channels)
#         # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
#
#     def forward(self, x):
#         """
#         sab: spatial attention block
#         cab: channel att ention block
#         :param x: input tensor
#         :return: sab + cab
#         """
#         sab = self.sab(x)
#         cab = self.cab(x)
#         out = sab + cab
#         return out
#
#
# class BA_CS_FRN_MISH_CCC_MSS_3CS_4NetV1(nn.Module):
#     def __init__(self, channels, classes, start_neurons=16):
#         """
#         :param classes: the object classes number.
#         :param channels: the channels of the input image.
#         """
#         super(BA_CS_FRN_MISH_CCC_MSS_3CS_4NetV1, self).__init__()
#         self.encoder0 = ResEncoder(channels, start_neurons * 2)
#         self.encoder1 = ResEncoder(start_neurons * 2, start_neurons * 4)
#         self.encoder2 = ResEncoder(start_neurons * 4, start_neurons * 8)
#         self.encoder3 = ResEncoder(start_neurons * 8, start_neurons * 16)
#         self.encoder4 = ResEncoder(start_neurons * 16, start_neurons * 32)
#         self.downsample = downsample()
#
#         self.affinity_attention = AffinityAttention(start_neurons * 32)
#         self.decoder4 = Decoder(start_neurons * 32, start_neurons * 16)
#         self.decoder3 = Decoder(start_neurons * 16, start_neurons * 8)
#         self.decoder2 = Decoder(start_neurons * 8, start_neurons * 4)
#         self.decoder1 = Decoder(start_neurons * 4, start_neurons * 2)
#         self.deconv40 = deconv0(start_neurons * 32, start_neurons * 16)
#         self.deconv30 = deconv0(start_neurons * 16, start_neurons * 8)
#         self.deconv20 = deconv0(start_neurons * 8, start_neurons * 4)
#         self.deconv10 = deconv0(start_neurons * 4, start_neurons * 2)
#
#         self.deconv41= deconv1(start_neurons * 32, start_neurons * 16)
#         self.deconv31 = deconv1(start_neurons * 16, start_neurons * 8)
#         self.deconv21 = deconv1(start_neurons * 8, start_neurons * 4)
#         self.deconv11 = deconv1(start_neurons * 4, start_neurons * 2)
#
#         self.final = nn.Conv2d(start_neurons * 2, classes, kernel_size=1)
#
#         self.mca = MCA(start_neurons * 2, ratio=16)
#
#         # self.conv1x1_4 = nn.Conv2d(start_neurons * 48, start_neurons * 32, 1)
#         # self.conv1x1_3 = nn.Conv2d(start_neurons * 24, start_neurons * 16, 1)
#         # self.conv1x1_2 = nn.Conv2d(start_neurons * 12, start_neurons * 8, 1)
#         # self.conv1x1_1 = nn.Conv2d(start_neurons * 6, start_neurons * 4, 1)
#         # self.conv1x1 = nn.Conv2d(start_neurons * 64, start_neurons * 32, 1)
#
#
#
#     def forward(self, x):
#         enc_input = self.encoder0(x)
#         down0 = self.downsample(enc_input)
#
#         enc1 = self.encoder1(down0)
#         down2 = self.downsample(enc1)
#
#         enc2 = self.encoder2(down2)
#         down3 = self.downsample(enc2)
#
#         enc3 = self.encoder3(down3)
#         down4 = self.downsample(enc3)
#
#         input_feature = self.encoder4(down4)
#
#         mca = self.mca(enc_input, enc1, enc2, enc3)
#
#         y = 0.2 * (mca.expand_as(input_feature)) + 0.8 * input_feature
#
#         attention = self.affinity_attention(y)
#         attention_fuse = input_feature + attention
#
#
#         attention_fuse1 = self.deconv40(attention_fuse)
#         attention_fuse1 = self.deconv30(attention_fuse1)
#         attention_fuse1 = self.deconv20(attention_fuse1)
#         attention_fuse1 = self.deconv10(attention_fuse1)
#         attention_fuse1 = self.final(attention_fuse1)
#         attention_fuse1 = nn.Sigmoid()(attention_fuse1)
#
#         up4 = self.deconv41(attention_fuse)
#         up4 = torch.cat([enc3, up4], dim=1)
#         dec4 = self.decoder4(up4)
#         dec41 = self.deconv30(dec4)
#         dec41 = self.deconv20(dec41)
#         dec41 = self.deconv10(dec41)
#         dec41 = self.final(dec41)
#         dec41 = nn.Sigmoid()(dec41)
#
#         up3 = self.deconv31(dec4)
#         up3 = torch.cat([enc2, up3], dim=1)
#         dec3 = self.decoder3(up3)
#         dec31 = self.deconv20(dec3)
#         dec31 = self.deconv10(dec31)
#         dec31 = self.final(dec31)
#         dec31 = nn.Sigmoid()(dec31)
#
#         up2 = self.deconv21(dec3)
#         up2 = torch.cat([enc1, up2], dim=1)
#         dec2 = self.decoder2(up2)
#         dec21 = self.deconv10(dec2)
#         dec21 = self.final(dec21)
#         dec21 = nn.Sigmoid()(dec21)
#
#         up1 = self.deconv11(dec2)
#         up1 = torch.cat([enc_input, up1], dim=1)
#         dec1 = self.decoder1(up1)
#         dec1 = self.final(dec1)
#         dec1 = nn.Sigmoid()(dec1)
#
#         final = attention_fuse1 * 0.1 + dec41 * 0.1 + dec31 * 0.1 + dec21 * 0.2 + dec1 * 0.5
#
#         return final
#
#
# if __name__ == '__main__':
#     net = BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU_7(1, 1).to(device)
#     data_arr = torch.rand(size=(1, 1, 256, 256)).to(device)
#     outputs = net(data_arr)
#     # outputs = net(data_arr)
#     print(outputs.size())
#

    
