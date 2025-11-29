import torch
import torch.nn as nn
import time
from fvcore.nn import FlopCountAnalysis
from torchinfo import summary  # 用于查看模型的Flops和内存
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 这里是你的网络定义
from only_for_vessel_seg.networks.comparison.CE_Net import CE_Net
from only_for_vessel_seg.networks.comparison.DU_Net import DU_Net
from only_for_vessel_seg.New_Net.Net_1.Discussions.U_Net import U_Net
from only_for_vessel_seg.New_Net.Net_1.Discussions.GT_DLA_dsHFF import GT_DLA_dsHFF
from only_for_vessel_seg.networks.common.CS_REAP_AB_test import CSNet_BR_RP_AB_1
from only_for_vessel_seg.New_Net.Net_1.UU_C_7 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU_7


# 自定义函数来测量推理时间
def measure_inference_time(model, input_tensor):
    start_time = time.time()
    model(input_tensor)
    end_time = time.time()
    return end_time - start_time


# 创建输入张量（假设输入尺寸是1x3x512x512）
input_tensor = torch.randn(1, 1, 512, 512).to(device)

# 创建模型列表
models = [
    U_Net(1, 1),
    CE_Net(1, 1),
    DU_Net(1, 1),
    CSNet_BR_RP_AB_1(1, 1),
    GT_DLA_dsHFF(1, 1),
    BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU_7(1, 1)
]
model_names = ["U-Net", "CE-Net", "DU-Net", "CSNet_BR_RP_AB_1", "GT_DLA_dsHFF", "BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU_7"]

# 创建一个字典来存储结果
results = {}

# 计算每个模型的Flops、内存和推理时间
for model, name in zip(models, model_names):
    model = model.to(device)  # 将模型移动到设备上
    print(f"Measuring for {name}...")

    # 使用torchinfo来显示模型的内存信息（注意这里直接通过summary获取）
    model_summary = summary(model, input_data=input_tensor)

    # 使用fvcore计算模型的Flops
    flops = FlopCountAnalysis(model, input_tensor).total()  # FLOPs

    # 测量推理时间
    reasoning_time = measure_inference_time(model, input_tensor)

    # 计算内存占用（这个仅为近似值，准确计算需要考虑每一层的激活值）
    memory_estimate = model_summary.total_params * 4 / 1e6  # 假设每个参数占4字节（32位浮点数）

    # 估算激活的内存占用
    activation_memory = input_tensor.size(1) * input_tensor.size(2) * input_tensor.size(
        3) * 4 * 2 / 1e6  # 每层占用的内存，假设每个值是32位

    total_memory = memory_estimate + activation_memory  # 总内存

    # 存储结果
    results[name] = {
        'Parameters': model_summary.total_params,  # 总参数量
        'Flops': flops,
        'Memory (MB)': total_memory,
        'Reasoning Time (s)': reasoning_time
    }

# 输出结果
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"  Parameters: {metrics['Parameters'] / 1e6}x10^6")
    print(f"  Flops: {metrics['Flops'] / 1e9}x10^9 GFLOPs")  # 转换为GFLOPs
    print(f"  Memory: {metrics['Memory (MB)']} MB")
    print(f"  Reasoning Time: {metrics['Reasoning Time (s)']} s\n")
