# CUDA 13 + cuDNN 9 环境安装参考

## 硬件要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | NVIDIA RTX 3060+ (6GB+) | RTX 4090 × 2 (24GB) |
| 内存 | 16 GB | 32 GB+ |
| 磁盘 | 50 GB 可用 | 200 GB+ |
| 系统 | Windows 10/11 64-bit | Windows 11 23H2+ |

## 安装步骤

### 1. NVIDIA 驱动

确保驱动版本 ≥ 610.x（for CUDA 13）：

```powershell
nvidia-smi
```

如果驱动过旧，从 https://www.nvidia.com/drivers 下载更新。

### 2. CUDA Toolkit 13.0

```powershell
# 下载: https://developer.nvidia.com/cuda-downloads
# 选择: Windows → x86_64 → 13.0 → exe(local)
# 安装后验证:
nvcc --version     # 应显示 V13.0.48
```

### 3. cuDNN 9.24 for CUDA 13

```powershell
# 下载: https://developer.nvidia.com/cudnn-downloads
# 选择: Windows → x86_64 → CUDA 13 → zip
# 解压后复制到 CUDA Toolkit 目录:
# bin/*.dll → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\
# include/* → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include\
# lib/*     → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\
```

### 4. Python 虚拟环境

```bash
# 创建环境
python -m venv D:\face_anon_env
D:\face_anon_env\Scripts\activate

# 安装核心依赖
pip install onnxruntime-gpu==1.27.0
pip install deface
pip install onnx
```

### 5. 确保 CUDA DLL 可被 onnxruntime 找到

```bash
# 将 CUDA 13 + cuDNN 9 DLL 复制到 onnxruntime capi 目录
cp "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\cudnn*.dll" \
   "D:\face_anon_env\Lib\site-packages\onnxruntime\capi\"
cp "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64\cublas64_13.dll" \
   "D:\face_anon_env\Lib\site-packages\onnxruntime\capi\"
cp "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64\cublasLt64_13.dll" \
   "D:\face_anon_env\Lib\site-packages\onnxruntime\capi\"
cp "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64\cudart64_13.dll" \
   "D:\face_anon_env\Lib\site-packages\onnxruntime\capi\"
```

### 6. 验证安装

```bash
D:\face_anon_env\Scripts\python scripts\audit_env.py
```

确保所有检查项均为 ✅。

## 版本兼容矩阵

```
Driver 610.x  ──▶  CUDA 13.0  ──▶  cuDNN 9.24
                                        │
                                        ▼
                              onnxruntime-gpu 1.27.0
                                    CUDA EP ✅
```
