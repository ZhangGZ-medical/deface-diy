# deface-diy

医疗视频面部脱敏管线 — CUDA 13 原生 + 双 RTX 4090 + NVENC 全速方案。

基于 [deface](https://github.com/ORB-h/deface) 1.5.0 + onnxruntime-gpu 1.27.0 + CUDA 13.0 + cuDNN 9.24。

## 功能

- 🎭 自动检测并模糊视频中的人脸
- 🔄 自动 MTS/AVI/MOV/MKV → MP4 转封装
- ⚡ 双 RTX 4090 CUDA 13 分片并行处理
- 🎬 NVENC 硬件编码加速
- 📁 单视频 / 批量文件夹两种模式

## 性能

| 方案 | 单卡 | 双卡 | 11分钟1080p视频 |
|------|------|------|:---:|
| CUDA 13 Native + NVENC | ~39 it/s | ~72 it/s | **~4.5 min** |

## 依赖

```
deface>=1.5.0
onnxruntime-gpu==1.27.0
Python>=3.10
CUDA Toolkit 13.0
cuDNN 9.24
NVIDIA Driver >=610
```

## 快速开始

```bash
# 1. 环境审查
D:\face_anon_env\Scripts\python scripts/audit_env.py

# 2. 单视频处理
D:\face_anon_env\Scripts\python scripts/deface_single.py video.MTS

# 3. 批量文件夹处理
D:\face_anon_env\Scripts\python scripts/deface_batch.py D:\videos

```

## 文件结构

```
deface-diy/
├── SKILL.md                       # WorkBuddy 技能定义
├── README.md                      # 本文件
├── scripts/
│   ├── audit_env.py               # 环境审查（GPU/CUDA/cuDNN/ort/deface/FFmpeg）
│   ├── deface_single.py           # 单视频双卡处理
│   └── deface_batch.py            # 批量文件夹双卡处理
└── references/
    ├── cuda13_install.md          # CUDA 13 + cuDNN 9 安装指南
    └── troubleshooting.md         # 常见问题排错
```

## 安全

完全离线运行，不上传任何数据。仅读取输入视频，写入 `_anonymized.mp4` 输出，不修改原始文件。

## 许可

本技能为 ZhangGZ-medical DIY 技能系列之一。
