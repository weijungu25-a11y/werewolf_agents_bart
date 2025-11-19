"""
utils.py
通用工具函数
"""

import os
import torch
import json
import numpy as np
import random

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_config(config, save_path):
    """保存配置文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_directory(directory):
    """创建目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def print_device_info():
    """打印设备信息"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"可用GPU内存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

def calculate_statistics(data, name):
    """计算统计信息"""
    if not data:
        return f"{name}: 无数据"
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    max_val = np.max(data)
    min_val = np.min(data)
    
    return f"{name}: 均值={mean_val:.4f}, 标准差={std_val:.4f}, 最大值={max_val:.4f}, 最小值={min_val:.4f}"