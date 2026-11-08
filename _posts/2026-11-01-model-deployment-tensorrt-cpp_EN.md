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

I have recently been learning model deployment and, following [this blog post](https://www.cnblogs.com/CrescentWind/p/18229812), implemented a simple deployment workflow, available in [this repository](https://github.com/Amroning26/trt-cpp-starter), to help me understand the process. I am recording what I have learned in this article for future reference.



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

# Preparing the Files

- Since the purpose is to learn about model deployment rather than model training, I downloaded the `mnist.onnx` model provided by Microsoft's `onnxjs-demo` project:

```
https://github.com/microsoft/onnxjs-demo/raw/master/public/mnist.onnx
```

- The digit image used for testing should resemble the MNIST dataset as closely as possible: a white digit centered on a black background.

![]({{ '/assets/images/num.jpg' | relative_url }})

# The Code

First, here is an overview of the deployment and inference process (see [this blog post](https://www.cnblogs.com/shaoxx333/p/16630781.html)):

Model initialization -> Input preprocessing -> Copy data from host memory to CUDA device memory -> Inference -> Copy inference results from CUDA device memory back to host memory -> Postprocessing

## Model Initialization

This part corresponds to the following code in `main.cpp`:

```cpp
        EngineManager manager("../mnist.onnx", "../mnist.engine");

        if (!manager.loadOrBuild()) {
            std::cerr << "Failed to load or build engine\n";
            return 1;
        }
```

- According to the official description, ONNX is an open format for representing machine learning models. It describes a model's computation graph, including its operators, weights, tensors, and the connections between them.

  I understand ONNX as a common format for saving a trained model so that it can be used in different environments without depending on PyTorch.

- An `.engine` file is the serialized result of building a model with TensorRT. TensorRT can deserialize it for use at runtime. During the build, TensorRT generates an execution plan based on the model and the build configuration.

- ONNX is the input to the engine build; the engine is the TensorRT execution artifact.

  Once the current program has built an engine, it creates an execution context from the in-memory engine and runs inference. On subsequent runs, it reads the engine data from the file, restores it with `deserializeCudaEngine`, and uses it in the same inference workflow.

- Why serialize the engine to a local `.engine` file?

  Building an engine requires parsing the ONNX model, optimizing it, and generating an execution plan, which takes longer than reading an existing engine. Saving it to disk allows subsequent launches to skip the build and reduces initialization time. Deployment can also use the engine file alone, avoiding the build time and some build dependencies on the target machine.

  An engine depends on the TensorRT version, GPU hardware, and build configuration. Unlike ONNX, it cannot be treated as a universal model file. In a different environment, an existing engine will often fail to load or no longer be suitable, so it must be rebuilt from ONNX.

The current code therefore works as follows:

1. First, try to deserialize an existing `.engine` file with `bool EngineManager::loadSerializedEngine()`.

   Some details of this function:

   - `std::ifstream file(engine_path_, std::ios::binary | std::ios::ate);`

     `std::ios::binary` opens the file in binary mode. The `.engine` file contains serialized bytes rather than text; binary mode avoids conversions such as newline translation on some systems.

     `std::ios::ate` moves the read position to the end immediately after opening the file ("ate" stands for "at end"). The subsequent `tellg()` call can then obtain the end position, which is the file size.

   - `const std::streamsize size = file.tellg();`

     `tellg()` returns the current read position. Since the position is at the end, this value is the file size in bytes. Its return type is `std::streampos`; here it is assigned to `std::streamsize` so the code can check whether the file is empty and allocate a buffer of the appropriate size.

   - `file.seekg(0, std::ios::beg);`

     `seekg` sets the read position. An offset of `0` relative to `std::ios::beg` moves it back to the first byte, allowing the subsequent `file.read(...)` to read the whole engine file into the buffer.

2. If the `.engine` file does not exist, build an engine starting from the ONNX model with `bool EngineManager::buildEngineFromOnnx()`.

   - This function initializes and builds a TensorRT engine from ONNX in the following steps:

     ```
     Create a Builder
         ↓
     Create a Network (the model's computation graph inside TensorRT)
         ↓
     Create an ONNX Parser and parse the ONNX model into the Network
         ↓
     Create a Config and set build options such as precision and workspace
         ↓
     Build an Engine from the Network and Config using the Builder
         ↓
     Serialize the Engine and save it to a file
     ```

   - The following is the old code; it has since been changed as described below:

     ```c++
         const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
     
         NetworkPtr network{builder->createNetworkV2(explicit_batch)};
     ```

     `createNetworkV2()` accepts a set of flags, with each bit representing an option.

     `kEXPLICIT_BATCH` is the bit index for this option, which is 0 here.

     `static_cast<uint32_t>(...)` converts the enum value to an integer to use as the shift count.

     `1U << ...` shifts the unsigned integer 1 left to the corresponding position, setting that bit.

`kEXPLICIT_BATCH` means the batch dimension is part of the tensor shape. For example, an image input shape is written as `[N, C, H, W]`. The older implicit batch mode handles the batch separately, so network shapes contain only `[C, H, W]`.

Starting with TensorRT 10.0, networks always use explicit batch, and this flag is deprecated and ignored. In TensorRT 10.9.0.34, `kEXPLICIT_BATCH` is still present in the API and can be compiled, but it is marked deprecated and produces a warning:

```bash
warning: ‘nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH’ is deprecated [-Wdeprecated-declarations]
```

This part has been updated to:

```c++
NetworkPtr network{builder->createNetworkV2(0U)};
```

- The following is also old code; it has since been changed as described below:

  ```c++
      if (builder->platformHasFastFp16()) {
          config->setFlag(nvinfer1::BuilderFlag::kFP16);
      }
  ```

  It first checks whether the platform has fast FP16 compute capability. If it does, TensorRT is allowed to select FP16 implementations during the build.

FP16 is a 16-bit floating-point format. Compared with FP32, it generally reduces device memory use and data transfer and can use GPU hardware for acceleration, but it also reduces numerical precision.

This permits FP16; it does not force every layer to use it. TensorRT chooses implementations based on operator support, performance, and constraints, so it does not guarantee a speedup for every model.

Newer versions of TensorRT are moving toward strongly typed networks, where tensor types in the model determine precision. `platformHasFastFp16()` is also deprecated. It is still available in the current version, but new code should not rely on it, and compiling produces a warning:

```bash
warning: ‘bool nvinfer1::IBuilder::platformHasFastFp16() const’ is deprecated [-Wdeprecated-declarations]
```

This part has been updated to:

```cpp
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }

    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return false;
    }

    if(properties.major > 5 || (properties.major == 5 && properties.minor >= 3)){
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
```

1. `cudaGetDevice()` obtains the current CUDA device number and writes it to `device`.
2. `cudaDeviceProp` stores device properties; `cudaGetDeviceProperties()` fills it using the device number.
3. `major` and `minor` are the GPU compute capability's major and minor version numbers. The condition sets TensorRT's `kFP16` build flag for compute capability 5.3 or higher.

- `config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,256ULL << 20);`

This sets the workspace memory pool limit to 256 MiB:

```cpp
256ULL << 20
// Equivalent to 256 × 2²⁰ bytes
// ULL denotes an unsigned long long integer
```

Workspace is temporary memory that operators may need during execution. This is neither a limit on the entire program's device memory nor an immediate allocation of 256 MiB. Model weights and input/output buffers have their own memory costs. The workspace limit constrains this kind of temporary memory and affects which execution strategies TensorRT can choose during the build.

## Input Preprocessing

Preprocessing converts raw data into the input expected by the model. For images, this usually involves resizing, handling color channels, converting the data type, and adjusting the value range, then arranging the tensor shape and data layout as required by the model. The exact steps depend on the preprocessing used during training and export and must also meet the engine's input requirements. If preprocessing during inference differs from training, predictions may degrade even when the engine runs successfully.

For image input, it helps to know that:

- An image can be viewed as an array:
  - A grayscale image is two-dimensional (height × width), with a brightness value at each position.
  - A color image generally has three channels, with three color components per pixel.
  - Some images have a fourth channel, usually alpha transparency.
- Common image tensor shapes include:
  - `NCHW`: batch, channels, height, width.
  - `NHWC`: batch, height, width, channels.
- OpenCV reads three-channel color images in BGR order by default. A single image is typically stored with interleaved channels in HWC layout, while some models require RGB order and NCHW input. Color channel order and tensor dimension order are different concepts and must be checked and converted separately.

The current preprocessing function is `cv::Mat process(const std::string& image_path) const`. Its steps are straightforward:

Read the image → convert it to grayscale according to its channel count → resize it to 28 × 28 → convert it to 32-bit floating point and multiply pixel values by 1/255.

There is also an unused function, `hwc2chw`, which converts image data from HWC to CHW layout. This example uses a single-channel image. With one channel, the two layouts have the same contiguous pixel order, so no rearrangement is needed. `process()` returns a single-channel `cv::Mat`; the batch and channel dimensions depend on how the model input is interpreted.

## Inference

The overall process is:

Initialize the inference runner -> Create a CUDA stream and allocate GPU input and output buffers -> Copy input from CPU memory to the GPU input buffer -> Run inference -> Asynchronously copy the GPU output buffer back to the CPU.

The runner is initialized with the previously created engine: `InferenceRunner runner(*manager.engine());`. Some points worth recording:

- `nvinfer1::Dims` is TensorRT's structure for representing tensor dimensions. Think of it as a fixed-capacity list of integers: `nbDims` is the number of dimensions, and `d[]` stores their sizes.

```cpp
    const nvinfer1::Dims input_dims = engine_.getTensorShape(input_name_);

    const nvinfer1::Dims output_dims = engine_.getTensorShape(output_name_);
```

`getTensorShape()` queries the engine for a tensor's shape by name and returns a `Dims`. Then `tensorSize()` iterates over the dimensions and multiplies them:

```cpp
for (int i = 0; i < dims.nbDims; ++i) {
    size *= dims.d[i];
}
```

This gives the total number of elements.

- Think of `ICudaEngine` as the model's execution plan and `IExecutionContext` as an instance that executes that plan with a particular set of inputs and runtime state.

  The engine holds the model information and execution strategy built by TensorRT. Inference also needs runtime state, such as input and output tensor addresses, dynamic input dimensions, and execution state. `IExecutionContext` handles these.

  The runner creates a context during construction:

  ```cpp
  std::unique_ptr<nvinfer1::IExecutionContext, TrtContextDeleter> context_;

  context_.reset(engine_.createExecutionContext());
  ```

  It then binds the GPU input and output buffers to the context:

  ```cpp
  context_->setTensorAddress(input_name_, device_input_.data());
  context_->setTensorAddress(output_name_, device_output_.data());
  ```

- `InferenceRunner::infer(const cv::Mat& input)` receives the preprocessed image, runs one TensorRT inference, and returns the model output as a `std::vector<float>`.

  - Asynchronously copy the input from CPU memory to the GPU:

    ```cpp
        checkCuda(
            cudaMemcpyAsync(device_input_.data(), input.ptr<float>(), input_size_ * sizeof(float), cudaMemcpyHostToDevice, stream_.get()), "cudaMemcpyAsync host to device");
    ```

    The arguments of `cudaMemcpyAsync` are:

    ```
    dst: Destination address, here the input buffer allocated on the GPU.
    src: Source address, here the start of the cv::Mat pixel data on the CPU.
    count: Number of bytes to copy.
    kind: Copy direction; cudaMemcpyHostToDevice means CPU host memory to GPU device memory.
    stream: The CUDA stream to which the copy is submitted.
    ```

    `cudaStream_t` is CUDA's stream handle type. A stream can be thought of as a GPU work queue: the CPU submits memory copies, kernels, or TensorRT inference to it, and the GPU executes operations in that stream in order.

  - Have TensorRT execute the engine:

    ```cpp
    context_->enqueueV3(stream_.get())
    ```

    `context_` executes the model using the input and output addresses bound during construction, submitting inference to the same stream.

  - Prepare a CPU output container and asynchronously copy the GPU output back:

    ```cpp
        std::vector<float> output(static_cast<std::size_t>(output_size_));

        checkCuda(
            cudaMemcpyAsync(output.data(), device_output_.data(), output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_.get()), "cudaMemcpyAsync device to host");
    ```

  - Wait for the stream to finish, then return the result:

    ```cpp
        checkCuda(cudaStreamSynchronize(stream_.get()), "cudaStreamSynchronize");
    
        return output;
    ```

    `cudaStreamSynchronize(stream)` waits for all previously submitted operations in the specified stream to finish.

## Postprocessing Inference Results

Postprocessing converts the `output` vector returned by TensorRT into two more interpretable results: the predicted class `label` and its corresponding `probability`.

One detail worth recording is how Softmax calculates the probability of the largest entry:

```cpp
    float max_value = output[best_index];
    float sum = 0.0f;

    for (float value : output) {
        sum += std::exp(value - max_value);
    }

    float probability = std::exp(output[best_index] - max_value) / sum;
```

$$
\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

`output` contains the model's scores, or logits, for each class. Subtracting the maximum before exponentiation leaves the mathematical result unchanged while reducing the risk of overflow. `best_index` points to the maximum, so the numerator is exp(0), or 1. Each term in the denominator represents a class's weight relative to the maximum score. In other words, the class scores are converted to positive weights, and the largest class's share of the total is calculated. Subtracting the maximum makes the calculation more stable without changing the final proportions.

Here is an example result:

![]({{ '/assets/images/result.jpg' | relative_url }})

