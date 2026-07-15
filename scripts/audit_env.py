#!/usr/bin/env python
"""
Deface 环境审查脚本 — 检查本机是否满足医疗视频面部脱敏管线的运行条件。
检查项: GPU, CUDA, cuDNN, onnxruntime, Python, FFmpeg, 磁盘, 内存
"""

import os, sys, ctypes, shutil, subprocess, json


def check_gpu():
    """检查 GPU 和 CUDA"""
    result = {"status": "unknown", "details": []}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            encoding="utf-8", timeout=15
        ).strip()
        gpus = [l.split(", ") for l in out.split("\n")]
        result["status"] = "pass" if gpus else "fail"
        for i, (name, mem, drv, cc) in enumerate(gpus):
            result["details"].append({
                "index": i, "name": name, "memory_mb": int(mem),
                "driver": drv, "compute_capability": cc
            })
        result["summary"] = f"{len(gpus)} GPU(s): " + ", ".join(
            f"{g['name']} ({g['memory_mb']/1024:.0f}GB)" for g in result["details"]
        )
    except Exception as e:
        result["status"] = "fail"
        result["summary"] = f"nvidia-smi 不可用 ({e})"
    return result


def check_cuda():
    """检查 CUDA Toolkit"""
    result = {"status": "unknown", "details": []}
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if not os.path.isdir(cuda_base):
        result["status"] = "fail"
        result["summary"] = "未找到 CUDA Toolkit"
        return result
    versions = os.listdir(cuda_base)
    for v in sorted(versions, reverse=True):
        vpath = os.path.join(cuda_base, v)
        nvcc = os.path.join(vpath, "bin", "nvcc.exe")
        cublas13 = os.path.join(vpath, "bin", "x64", "cublas64_13.dll")
        cublas12 = os.path.join(vpath, "bin", "cublas64_12.dll")

        entry = {
            "version": v,
            "path": vpath,
            "nvcc": os.path.exists(nvcc),
        }

        # Check cublas version hint
        dlls = []
        for root, dirs, files in os.walk(os.path.join(vpath, "bin")):
            for f in files:
                if "cublas" in f.lower() and f.endswith(".dll"):
                    dlls.append(f)
        entry["cublas_dlls"] = dlls[:3]
        result["details"].append(entry)

    result["status"] = "pass"
    result["summary"] = f"CUDA Toolkit: {', '.join(v for v in versions)}"
    return result


def check_cudnn():
    """检查 cuDNN — 在 CUDA Toolkit bin 中查找"""
    result = {"status": "unknown", "details": []}
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    for v in sorted(os.listdir(cuda_base), reverse=True):
        bin_dir = os.path.join(cuda_base, v, "bin")
        if not os.path.isdir(bin_dir):
            continue
        dlls = []
        for f in os.listdir(bin_dir):
            if "cudnn" in f.lower() and f.endswith(".dll"):
                dlls.append(f)
        if dlls:
            entry = {"cuda_version": v, "cuDNN_dlls": sorted(dlls)[:5],
                     "dll_count": len(dlls)}
            result["details"].append(entry)

    if result["details"]:
        result["status"] = "pass"
        dlls_found = result["details"][0].get("dll_count", 0)
        result["summary"] = f"cuDNN: {dlls_found} DLL(s) in CUDA Toolkit"
    else:
        result["status"] = "fail"
        result["summary"] = "未找到 cuDNN DLL"
    return result


def check_onnxruntime():
    """检查 onnxruntime 及其 provider"""
    result = {"status": "unknown", "details": {}}
    try:
        import onnxruntime as ort
        result["details"]["version"] = ort.__version__
        providers = ort.get_available_providers()
        result["details"]["providers"] = providers
        has_cuda = "CUDAExecutionProvider" in providers
        has_trt = "TensorrtExecutionProvider" in providers
        result["status"] = "pass" if has_cuda else "warn"
        result["summary"] = "onnxruntime-gpu " + ort.__version__
        if has_cuda:
            result["summary"] += " | CUDA EP ✅"
        if has_trt:
            result["summary"] += " | TensorRT EP ✅"
    except ImportError:
        result["status"] = "fail"
        result["summary"] = "onnxruntime-gpu 未安装"
    except Exception as e:
        result["status"] = "fail"
        result["summary"] = f"onnxruntime 错误: {e}"
    return result


def check_python():
    return {"status": "pass", "summary": f"Python {sys.version.split()[0]}",
            "details": {"version": sys.version.split()[0], "executable": sys.executable}}


def check_ffmpeg():
    try:
        out = subprocess.check_output(["ffmpeg", "-version"], encoding="utf-8",
                                      stderr=subprocess.STDOUT, timeout=10)
        nvenc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        has_nvenc = "h264_nvenc" in nvenc.stdout
        line = out.split("\n")[0]
        return {"status": "pass", "summary": f"{line.strip()} | NVENC: {'✅' if has_nvenc else '❌'}",
                "details": {"version_line": line.strip(), "nvenc": has_nvenc}}
    except Exception as e:
        return {"status": "fail", "summary": f"FFmpeg 不可用: {e}", "details": {}}


def check_disk():
    usage = shutil.disk_usage("C:\\")
    free_gb = usage.free / (1024**3)
    return {"status": "pass" if free_gb > 50 else "warn",
            "summary": f"C: 可用 {free_gb:.0f} GB",
            "details": {"free_gb": round(free_gb, 1), "total_gb": round(usage.total/(1024**3),1)}}


def check_ram():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)
        ]
    mem = MEMORYSTATUSEX()
    mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    total = mem.ullTotalPhys / (1024**3)
    avail = mem.ullAvailPhys / (1024**3)
    return {"status": "pass" if total >= 16 else "warn",
            "summary": f"{total:.0f} GB (可用 {avail:.0f} GB)",
            "details": {"total_gb": round(total,0), "available_gb": round(avail,0)}}


def check_deface():
    try:
        import deface
        v = getattr(deface, "__version__", "installed")
        return {"status": "pass", "summary": f"deface {v}", "details": {"version": v}}
    except ImportError:
        return {"status": "warn", "summary": "deface 未安装"}


def main():
    checks = {
        "GPU/CUDA 驱动": check_gpu(),
        "CUDA Toolkit": check_cuda(),
        "cuDNN": check_cudnn(),
        "onnxruntime": check_onnxruntime(),
        "deface": check_deface(),
        "Python": check_python(),
        "FFmpeg": check_ffmpeg(),
        "磁盘": check_disk(),
        "内存": check_ram(),
    }

    print("=" * 60)
    print("  医疗视频面部脱敏管线 — 环境审查")
    print("=" * 60)
    print()

    all_pass = True
    for name, check in checks.items():
        icon = "✅" if check["status"] == "pass" else "⚠️" if check["status"] == "warn" else "❌"
        print(f"  {icon} {name}: {check['summary']}")
        if check["status"] == "fail":
            all_pass = False

    print()
    if all_pass:
        print("✅ 所有检查通过，可以运行 deface 管线。")
    else:
        print("⚠️ 部分检查未通过，请先修复以上 ❌ 项。")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
