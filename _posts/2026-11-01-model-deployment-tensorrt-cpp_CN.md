---
layout: post
title: "模型部署简单示例（TensorRT C++）"
date: 2026-11-01 00:00:00 +0800
lang: zh-CN
categories:
  - 模型部署
tags:
  - TensorRT
  - C++
permalink: /miscellaneous/model-deployment/tensorrt-cpp/
---

{% include language-switch.html %}

最近在学习模型部署，实现了一个简易的模型部署流程，位于 [该仓库](https://github.com/Amroning26/trt-cpp-starter)  中，用于理解。为了不遗忘，在该篇文章中记录学到的东西。

先记录一下环境配置过程，然后再介绍代码内容。



# 环境配置

我的环境是 WSL，Ubuntu 24.04，Windows 侧已支持 WSL CUDA 的 NVIDIA 驱动。WSL 侧需要安装：

```
OpenCV C++ 开发包
CUDA Toolkit
TensorRT Runtime、开发头文件、ONNX Parser
TensorRT Samples 中的 logger.h 和 logger.cpp
```

需要 logger.h 和 logger.cpp， 是因为构建 Runtime、OnnxParser 等工具的时候，需要传入 `nvinfer1::ILogger &logger`，是一个抽象接口，要求提供 `log()` 实现。TensorRT 没有一个固定的日志类，因为日志应该由开发者决定如何处理，可以自己根据 `nvinfer1::ILogger &logger`自己实现一个日志类，我这里就从简使用 TensorRT Samples 中的示例日志类了



- 安装基础工具和 OpenCV 

```bash
sudo apt update
sudo apt install -y build-essential g++ cmake pkg-config git wget curl unzip libopencv-dev
```



确认 OpenCV 安装成功，能看到版本号

```bash
pkg-config --modversion opencv4
```



-  安装 CUDA Toolkit

添加 CUDA 官方源：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```



安装 CUDA 12.8 Toolkit ：

```bash
sudo apt install -y cuda-toolkit-12-8
```



设置环境变量：

```bash
cat >> ~/.bashrc <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
EOF

source ~/.bashrc
nvcc --version
```

看到 nvcc 信息，这部分就算成功了



- 安装 TensorRT

```bash
sudo apt install tensorrt=10.9.0.34-1+cuda12.8 tensorrt-dev=10.9.0.34-1+cuda12.8 libnvinfer-samples=10.9.0.34-1+cuda12.8
```

目前代码中有一些旧接口，为了不报错先装了 TensorRT 10.9 的版本。以后优化代码的的时候再去掉旧接口。



确认版本：

```bash
dpkg-query -W -f='${Package} ${Version}\n' tensorrt tensorrt-dev libnvinfer-dev libnvonnxparsers-dev
```



如果不想自己实现日志类，就可以和我一样下载 TensorRT Samples 的来用：

```bash
sudo git clone --depth 1 https://github.com/NVIDIA/TensorRT.git /usr/src/tensorrt
```

