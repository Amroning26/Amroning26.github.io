---
layout: post
title: "Simple Model Deployment Example (TensorRT C++)"
date: 2026-11-01 00:00:00 +0800
lang: en
categories:
  - 模型部署
tags:
  - TensorRT
  - C++
permalink: /en/miscellaneous/model-deployment/tensorrt-cpp/
---

{% include language-switch.html %}

I have recently been learning model deployment and implemented a simple deployment workflow, available in [this repository](https://github.com/Amroning26/trt-cpp-starter), to help me understand the process. I am recording what I have learned in this article for future reference.



I will first document the environment setup process, followed by an introduction to the code.



# Environment Setup

My environment is WSL running Ubuntu 24.04. The NVIDIA driver on Windows already supports CUDA in WSL. The following components need to be installed inside WSL:

```
OpenCV development packages for C++
CUDA Toolkit
TensorRT Runtime, development headers, and the ONNX Parser
logger.h and logger.cpp from the TensorRT Samples
```

We need `logger.h` and `logger.cpp` because building components such as the Runtime and `OnnxParser` requires passing an `nvinfer1::ILogger &logger` object. This is an abstract interface, so a `log()` implementation must be provided. TensorRT does not provide a fixed logging class because developers should decide how log messages are handled. You can implement your own logging class based on `nvinfer1::ILogger &logger`. For simplicity, I use the example logger class from the TensorRT Samples.



- Installing Basic Tools and OpenCV

```bash
sudo apt update
sudo apt install -y build-essential g++ cmake pkg-config git wget curl unzip libopencv-dev
```



To verify that OpenCV was installed successfully, check whether its version number is displayed.

```bash
pkg-config --modversion opencv4
```



-  Installing the CUDA Toolkit

Add the official CUDA repository:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```



Install CUDA Toolkit 12.8:

```bash
sudo apt install -y cuda-toolkit-12-8
```



Set the environment variables:

```bash
cat >> ~/.bashrc <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
EOF

source ~/.bashrc
nvcc --version
```

If information about `nvcc` is displayed, this part of the installation was successful.



- Installing TensorRT

```bash
sudo apt install tensorrt=10.9.0.34-1+cuda12.8 tensorrt-dev=10.9.0.34-1+cuda12.8 libnvinfer-samples=10.9.0.34-1+cuda12.8
```

The current code still uses some deprecated APIs, so I installed TensorRT 10.9 first to avoid compilation errors. I will remove the deprecated APIs when the code is improved later.



Verify the installed versions:

```bash
dpkg-query -W -f='${Package} ${Version}\n' tensorrt tensorrt-dev libnvinfer-dev libnvonnxparsers-dev
```



If you do not want to implement a logging class yourself, you can download and use the one from the TensorRT Samples, as I did.

```bash
sudo git clone --depth 1 https://github.com/NVIDIA/TensorRT.git /usr/src/tensorrt
```

