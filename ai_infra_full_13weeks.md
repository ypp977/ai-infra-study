

# ===== AI_INFRA_WEEK1 =====

# AI Infra 学习计划 - 第1周

## **Day 1 - Docker & GPU 容器化**

### 🎯 学习目标
- 掌握 Docker 基础命令与容器化流程  
- 能够构建 PyTorch 推理环境的 Docker 镜像  
- 在容器中运行 GPU 程序（使用 nvidia-docker）  

### 📌 学习重点
- Docker 基础命令：`run`、`ps`、`exec`、`logs`、`rm`  
- Dockerfile 构建镜像  
- 容器挂载卷、网络配置  
- `nvidia-docker` GPU 加速容器  

### ⏱ 时间安排
- 3h：Docker 基础 & 常用命令实验  
- 2h：写 Dockerfile 打包 PyTorch 环境  
- 2h：安装 nvidia-docker，运行 GPU 容器  
- 1h：实验总结（写 `report_day1.md`）  
- 2h：LeetCode（数组 + 哈希）  

### 🖥️ 命令示例
```bash
docker pull pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
docker run --gpus all -it --rm pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime /bin/bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 📂 交付成果
- 一个能运行 GPU 的 Docker 容器  
- Dockerfile 文件（安装 PyTorch + ONNX + TensorRT SDK）  
- 学习笔记：`report_day1.md`  

---

## **Day 2 - Kubernetes 基础 & GPU Pod**

### 🎯 学习目标
- 掌握 Kubernetes 基础对象：Pod、Deployment、Service  
- 能够在 K8s 集群中运行 GPU Pod  
- 部署一个 PyTorch 容器到 K8s  

### 📌 学习重点
- K8s 基本命令：`kubectl apply`、`get`、`describe`、`logs`  
- GPU device plugin（nvidia/k8s-device-plugin）  
- K8s Service 暴露模型服务  

### ⏱ 时间安排
- 3h：安装 Minikube/Kind，学习基本 kubectl 命令  
- 3h：部署 GPU Pod（带 nvidia-device-plugin）  
- 2h：使用 Service 暴露 GPU Pod  
- 2h：LeetCode（栈 + 队列）  

### 🖥️ 命令示例
```bash
minikube start --driver=docker --gpus=all
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
kubectl create deployment gpu-test --image=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
kubectl expose deployment gpu-test --type=NodePort --port=8888
```

### 📂 交付成果
- `k8s-gpu-demo.yaml`：GPU Pod 配置文件  
- GPU Pod 成功运行截图（`kubectl get pods`）  
- 学习笔记：`report_day2.md`  

---

## **Day 3 - PyTorch → ONNX**

### 🎯 学习目标
- 训练一个简单的 PyTorch 模型（MNIST/CIFAR10）  
- 导出 ONNX 模型，并使用 ONNX Runtime 进行推理  

### 📌 学习重点
- PyTorch 模型定义与保存  
- ONNX 导出与推理  
- K8s 部署 ONNX Runtime  

### ⏱ 时间安排
- 4h：PyTorch 训练 + 保存模型  
- 2h：导出 ONNX & 测试推理  
- 2h：写 K8s 部署文件（onnxruntime 容器）  
- 2h：LeetCode（链表 + 二分查找）  

### 🖥️ 代码示例
```python
import torch, torch.nn as nn
model = nn.Linear(784, 10)
dummy_input = torch.randn(1, 784)
torch.onnx.export(model, dummy_input, "mlp.onnx")
```

### 📂 交付成果
- `mlp.onnx`  
- `onnx_infer.py`  
- 学习笔记：`report_day3.md`  

---

## **Day 4 - TensorRT 优化**

### 🎯 学习目标
- 学习 TensorRT 基本用法  
- 将 ONNX 模型转换为 TensorRT engine  
- 使用 FP16/INT8 加速推理  

### 📌 学习重点
- `trtexec` 工具  
- TensorRT Python API  
- INT8/FP16 量化原理  

### ⏱ 时间安排
- 3h：安装 TensorRT & 环境配置  
- 3h：ONNX → TensorRT Engine  
- 2h：写推理脚本（TensorRT Python API）  
- 2h：LeetCode（双指针 + 滑动窗口）  

### 🖥️ 命令示例
```bash
trtexec --onnx=mlp.onnx --saveEngine=mlp_fp16.engine --fp16
```

### 📂 交付成果
- `mlp_fp16.engine`  
- `tensorrt_infer.py`  
- 学习笔记：`report_day4.md`  

---

## **Day 5 - Triton Inference Server**

### 🎯 学习目标
- 使用 Triton 部署 ONNX/TensorRT 模型  
- 学习 Triton 模型仓库结构  
- 使用 HTTP/gRPC Client 调用推理  

### 📌 学习重点
- Triton 模型部署目录  
- HTTP/gRPC Client SDK  
- Triton Metrics  

### ⏱ 时间安排
- 3h：拉取 tritonserver 镜像 & 运行  
- 3h：部署 ONNX 模型到 Triton  
- 2h：写 client 请求代码（Python）  
- 2h：LeetCode（DFS + BFS）  

### 🖥️ 命令示例
```bash
docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002   -v $PWD/models:/models nvcr.io/nvidia/tritonserver:23.05-py3 tritonserver --model-repository=/models
```

### 📂 交付成果
- `triton-deploy.yaml`  
- `client_infer.py`  
- 学习笔记：`report_day5.md`  

---

## **Day 6 - 监控体系（Prometheus + Grafana）**

### 🎯 学习目标
- 安装并配置 Prometheus + Grafana  
- 接入 Triton metrics  
- 制作 GPU Util/QPS/延迟 Dashboard  

### 📌 学习重点
- Prometheus 抓取 Triton metrics  
- Grafana Dashboard 配置  
- 基本告警机制  

### ⏱ 时间安排
- 3h：安装 Prometheus  
- 3h：安装 Grafana  
- 2h：制作 Dashboard  
- 2h：LeetCode（堆 + 优先队列）  

### 🖥️ 命令示例
```yaml
scrape_configs:
  - job_name: 'triton'
    static_configs:
      - targets: ['localhost:8002']
```

### 📂 交付成果
- Grafana Dashboard JSON  
- `prometheus.yaml`  
- 学习笔记：`report_day6.md`  

---

## **Day 7 - 项目整合**

### 🎯 学习目标
- 整合前6天的成果，形成第一个企业级 AI 推理平台雏形  
- 输出架构图 + README  

### 📌 学习重点
- 项目结构化整理  
- 架构文档编写  
- 简历亮点总结  

### ⏱ 时间安排
- 3h：代码整理  
- 3h：绘制架构图（Docker + K8s + Triton + Prometheus + Grafana）  
- 2h：编写 `README.md`  
- 2h：LeetCode（综合题目）  

### 📂 交付成果
- `ai-infra-inference-service/`
  - `Dockerfile`
  - `k8s/`
  - `onnx/`
  - `tensorrt/`
  - `prometheus/`
  - `grafana/`
  - `README.md`
- 架构图（PNG/SVG）  
- 学习笔记：`report_day7.md`  




# ===== AI_INFRA_WEEK2 =====

# AI Infra 学习计划 - 第2周

## **Day 8 - CUDA 基础**

### 🎯 学习目标
- 理解 CUDA 编程模型（thread, block, grid）
- 能够编写简单的 CUDA 程序（向量加法）
- 配置 nvcc 编译环境

### 📌 学习重点
- CUDA kernel 函数定义与调用
- 线程层次结构（grid, block, thread）
- CPU 与 GPU 内存交互

### ⏱ 时间安排
- 3h：CUDA 基础语法学习
- 2h：配置 CUDA 环境，运行 Hello World
- 3h：编写向量加法 CUDA 程序
- 2h：LeetCode（数组 & 前缀和）

### 🖥️ 代码示例
```cpp
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

### 📂 交付成果
- `cuda_hello.cu`
- `vector_add.cu`
- 学习笔记：`report_day8.md`

---

## **Day 9 - CUDA 内存管理**

### 🎯 学习目标
- 掌握 CUDA 内存分配与拷贝
- 理解 global / shared / constant 内存
- 实现矩阵加法 CUDA 版本

### 📌 学习重点
- `cudaMalloc`, `cudaMemcpy`
- global 与 shared memory 区别
- 性能对比实验

### ⏱ 时间安排
- 3h：CUDA 内存管理学习
- 3h：矩阵加法 CUDA 实现
- 2h：性能对比实验（CPU vs GPU）
- 2h：LeetCode（链表 & 快慢指针）

### 📂 交付成果
- `matrix_add.cu`
- 实验报告：`report_day9.md`

---

## **Day 10 - CUDA 优化技巧**

### 🎯 学习目标
- 理解 warp divergence 与 memory coalescing
- 学会使用 shared memory 优化矩阵乘法
- 分析 GPU 线程调度与性能瓶颈

### 📌 学习重点
- warp divergence 原理
- 内存访问模式优化
- 矩阵乘法性能优化

### ⏱ 时间安排
- 4h：学习优化技巧
- 3h：实现矩阵乘法 CUDA 程序
- 1h：性能测试与记录
- 2h：LeetCode（二分搜索 & 贪心）

### 📂 交付成果
- `matrix_mul.cu`
- 性能测试结果
- 学习笔记：`report_day10.md`

---

## **Day 11 - CUDA 实战项目**

### 🎯 学习目标
- 综合 CUDA 编程，完成矩阵乘法优化
- 对比 CPU vs GPU 性能差异
- 撰写实验报告

### 📌 学习重点
- Kernel 调优
- 使用 shared memory 减少 global memory 访问
- Block/Grid 大小调优

### ⏱ 时间安排
- 5h：矩阵乘法实现与优化
- 3h：性能测试
- 2h：LeetCode（动态规划入门）

### 📂 交付成果
- `matrix_mul_optimized.cu`
- 实验报告：`report_day11.md`

---

## **Day 12 - TensorRT Plugin 入门**

### 🎯 学习目标
- 学习 TensorRT Plugin 开发流程
- 编写一个简单的 Plugin（如自定义激活函数）
- 使用 TensorRT Plugin 加载模型

### 📌 学习重点
- TensorRT Plugin API
- C++/CUDA Plugin 实现
- Plugin 调试与集成

### ⏱ 时间安排
- 3h：学习 Plugin 开发文档
- 3h：编写简单 Plugin
- 2h：集成 Plugin 到 TensorRT
- 2h：LeetCode（字符串处理）

### 📂 交付成果
- `custom_plugin.cpp`
- `plugin_demo.py`
- 学习笔记：`report_day12.md`

---

## **Day 13 - Prometheus 深入**

### 🎯 学习目标
- 掌握自定义 Prometheus Exporter 编写
- 实现 GPU 温度采集并上报
- 接入 Grafana 展示

### 📌 学习重点
- Prometheus Python Client
- Exporter API
- Grafana Dashboard 配置

### ⏱ 时间安排
- 4h：编写 GPU Exporter
- 2h：Prometheus 集成测试
- 2h：Grafana Dashboard 配置
- 2h：LeetCode（区间问题）

### 🖥️ 代码示例
```python
from prometheus_client import Gauge, start_http_server
import random, time

gpu_temp = Gauge('gpu_temperature_celsius', 'GPU Temperature')

if __name__ == '__main__':
    start_http_server(8003)
    while True:
        gpu_temp.set(random.uniform(40, 80))
        time.sleep(5)
```

### 📂 交付成果
- `gpu_exporter.py`
- Grafana Dashboard JSON
- 学习笔记：`report_day13.md`

---

## **Day 14 - 周总结与项目增强**

### 🎯 学习目标
- 总结前两周成果，整理笔记
- 增强 AI 推理平台：增加监控告警
- 输出简历可用的项目描述

### 📌 学习重点
- 项目整合
- 架构文档撰写
- 简历亮点提炼

### ⏱ 时间安排
- 3h：整理代码与文档
- 3h：增加 Prometheus 告警规则
- 2h：绘制项目架构图
- 2h：LeetCode（综合题目）

### 📂 交付成果
- 更新后的 `ai-infra-inference-service/`
- 完整项目 README.md
- 架构图（SVG/PNG）
- 学习笔记：`report_day14.md`




# ===== AI_INFRA_WEEK3 =====

# AI Infra 学习计划 - 第3周

## **Day 15 - Kubernetes Operator 入门**

### 🎯 学习目标
- 学习 Kubernetes Operator 的基本概念
- 使用 Kubebuilder 编写一个简单的 Operator
- 理解 CRD（Custom Resource Definition）

### 📌 学习重点
- Operator 模式与 K8s 控制器
- Kubebuilder 框架
- CRD 定义与应用

### ⏱ 时间安排
- 3h：学习 Operator 基础概念
- 3h：安装并使用 Kubebuilder
- 2h：编写并运行一个简单的 CRD
- 2h：LeetCode（栈 + 队列）

### 📂 交付成果
- `sample-operator/`
- 自定义 CRD YAML
- 学习笔记：`report_day15.md`

---

## **Day 16 - Triton + Redis 缓存**

### 🎯 学习目标
- 在 Triton 推理服务中引入 Redis 缓存
- 减少重复推理请求的延迟
- 实现简单的 Key-Value 缓存逻辑

### 📌 学习重点
- Triton Python Client 与 Redis 交互
- 请求缓存设计
- 缓存命中率分析

### ⏱ 时间安排
- 3h：学习 Redis 基本用法
- 3h：在 Triton Client 中增加 Redis 缓存
- 2h：测试缓存效果
- 2h：LeetCode（哈希表）

### 🖥️ 代码示例
```python
import redis
import tritonclient.http as http

r = redis.Redis(host='localhost', port=6379)
client = http.InferenceServerClient(url="localhost:8000")

key = "input_123"
if r.exists(key):
    result = r.get(key)
else:
    result = client.infer("model_name", inputs=[...])
    r.set(key, result)
```

### 📂 交付成果
- `triton_redis_client.py`
- 缓存实验报告
- 学习笔记：`report_day16.md`

---

## **Day 17 - CUDA 优化进阶**

### 🎯 学习目标
- 深入理解 warp divergence 和 bank conflict
- 掌握 CUDA 优化工具（nvprof / Nsight）
- 实现共享内存优化版本的矩阵乘法

### 📌 学习重点
- Warp divergence 代码示例
- Bank conflict 避免策略
- CUDA 性能分析工具

### ⏱ 时间安排
- 4h：CUDA 优化理论学习
- 3h：编写优化代码
- 1h：性能分析报告
- 2h：LeetCode（双指针 + 滑动窗口）

### 📂 交付成果
- `matrix_mul_shared.cu`
- Nsight 截图/报告
- 学习笔记：`report_day17.md`

---

## **Day 18 - ONNX Runtime 高级优化**

### 🎯 学习目标
- 学习 ONNX Runtime Execution Provider
- 使用 TensorRT Execution Provider 加速推理
- 比较 CPU vs GPU vs TensorRT 性能

### 📌 学习重点
- ONNX Runtime EP 机制
- TensorRT EP 配置
- 性能对比实验

### ⏱ 时间安排
- 3h：ONNX Runtime 高级文档学习
- 3h：配置 TensorRT EP
- 2h：运行对比实验
- 2h：LeetCode（二分查找 + 动态规划）

### 📂 交付成果
- `onnxruntime_trt.py`
- 性能对比表格
- 学习笔记：`report_day18.md`

---

## **Day 19 - BERT 模型推理部署**

### 🎯 学习目标
- 下载并部署 BERT 模型
- 将 BERT 导出为 ONNX/TensorRT
- 使用 Triton Inference Server 部署 BERT

### 📌 学习重点
- HuggingFace Transformers
- BERT ONNX 导出
- Triton 部署大型模型

### ⏱ 时间安排
- 4h：BERT 模型准备
- 3h：导出 ONNX/TensorRT
- 1h：Triton 部署测试
- 2h：LeetCode（字符串处理）

### 📂 交付成果
- `bert.onnx`
- `bert_fp16.engine`
- `bert_triton_deploy.yaml`
- 学习笔记：`report_day19.md`

---

## **Day 20 - K8s HPA 自动扩缩容**

### 🎯 学习目标
- 学习 Kubernetes HPA（Horizontal Pod Autoscaler）
- 使用 Prometheus Adapter 进行自定义指标扩缩容
- 实现 Triton QPS 自动扩缩容

### 📌 学习重点
- HPA 基础
- Prometheus Adapter 配置
- Triton QPS 指标接入

### ⏱ 时间安排
- 3h：HPA 学习与实验
- 3h：Prometheus Adapter 部署
- 2h：HPA + Triton 实战
- 2h：LeetCode（区间问题）

### 📂 交付成果
- `hpa-triton.yaml`
- Prometheus Adapter 配置文件
- 学习笔记：`report_day20.md`

---

## **Day 21 - 第三周总结**

### 🎯 学习目标
- 整理第三周学习成果
- 输出项目增强版（支持缓存 + BERT + HPA）
- 更新简历描述

### 📌 学习重点
- 项目整合
- 性能优化总结
- 简历亮点提炼

### ⏱ 时间安排
- 3h：整理代码与文档
- 3h：更新项目架构图
- 2h：撰写简历亮点描述
- 2h：LeetCode（综合题目）

### 📂 交付成果
- 增强版 `ai-infra-inference-service/`
- 架构图（带缓存与 HPA）
- 项目总结文档
- 学习笔记：`report_day21.md`




# ===== AI_INFRA_WEEK4 =====

# AI Infra 学习计划 - 第4周

## **Day 22 - 高频面试题准备（容器与集群）**

### 🎯 学习目标
- 梳理 Docker 与 Kubernetes 高频面试题
- 掌握 K8s 调度、网络、存储相关知识点
- 编写简要问答文档

### 📌 学习重点
- Docker 与 K8s 区别
- Pod 调度策略
- K8s 网络模型与 CNI 插件

### ⏱ 时间安排
- 3h：学习 Docker vs K8s 相关知识
- 3h：整理调度与网络面试题
- 2h：编写问答文档
- 2h：LeetCode（数组 & 二分）

### 📂 交付成果
- `interview_docker_k8s.md`
- 学习笔记：`report_day22.md`

---

## **Day 23 - CUDA 面试题准备**

### 🎯 学习目标
- 梳理 CUDA 高频面试题
- 熟悉 warp divergence、bank conflict、memory coalescing 等概念
- 准备实际代码示例

### 📌 学习重点
- GPU 线程调度
- 内存访问优化
- CUDA 性能调优策略

### ⏱ 时间安排
- 4h：CUDA 高频考点学习
- 3h：准备示例代码（matrix_mul 优化版）
- 1h：编写问答文档
- 2h：LeetCode（动态规划）

### 📂 交付成果
- `interview_cuda.md`
- CUDA 示例代码
- 学习笔记：`report_day23.md`

---

## **Day 24 - TensorRT 面试题准备**

### 🎯 学习目标
- 梳理 TensorRT 优化相关的面试题
- 熟悉 FP16/INT8 量化原理
- 能够解释 TensorRT engine 工作流程

### 📌 学习重点
- TensorRT 架构与执行引擎
- 量化与校准
- TensorRT Plugin

### ⏱ 时间安排
- 3h：学习 TensorRT 官方文档
- 3h：整理面试题与答案
- 2h：准备 FP16/INT8 优化示例
- 2h：LeetCode（字符串）

### 📂 交付成果
- `interview_tensorrt.md`
- TensorRT 优化示例
- 学习笔记：`report_day24.md`

---

## **Day 25 - Triton 面试题准备**

### 🎯 学习目标
- 熟悉 Triton Inference Server 架构
- 掌握 Ensemble 模型与 Model Repository 概念
- 梳理 Triton 面试题

### 📌 学习重点
- Triton 模型仓库结构
- Ensemble 模型推理
- Triton Metrics

### ⏱ 时间安排
- 3h：Triton 文档学习
- 3h：整理面试题与答案
- 2h：准备 Triton 部署示例
- 2h：LeetCode（堆 & 优先队列）

### 📂 交付成果
- `interview_triton.md`
- Triton 示例配置
- 学习笔记：`report_day25.md`

---

## **Day 26 - Prometheus/Grafana 面试题准备**

### 🎯 学习目标
- 梳理 Prometheus 与 Grafana 面试题
- 掌握数据采集、报警机制
- 能够解释监控体系设计

### 📌 学习重点
- Prometheus 数据模型
- Exporter 编写
- Grafana Dashboard 与报警

### ⏱ 时间安排
- 3h：Prometheus 文档学习
- 3h：整理问答文档
- 2h：配置告警规则
- 2h：LeetCode（图论 BFS/DFS）

### 📂 交付成果
- `interview_prometheus_grafana.md`
- 告警配置文件
- 学习笔记：`report_day26.md`

---

## **Day 27 - 项目答辩准备**

### 🎯 学习目标
- 准备项目讲解（背景、架构、难点、亮点）
- 总结 20 个常见问答
- 撰写答辩稿

### 📌 学习重点
- 项目背景介绍
- 技术难点与解决方案
- 面试亮点提炼

### ⏱ 时间安排
- 4h：整理项目背景 & 架构图
- 3h：撰写常见问答
- 1h：编写答辩稿
- 2h：LeetCode（综合题）

### 📂 交付成果
- `project_pitch.md`
- 项目答辩文档
- 学习笔记：`report_day27.md`

---

## **Day 28 - 模拟面试**

### 🎯 学习目标
- 进行一次完整的模拟面试
- 包含算法、系统设计、项目讲解
- 复盘并优化回答

### 📌 学习重点
- 算法现场手写
- 系统设计（AI 推理平台）
- 项目答辩

### ⏱ 时间安排
- 3h：算法题模拟
- 3h：系统设计模拟
- 2h：项目讲解模拟
- 2h：复盘与总结

### 📂 交付成果
- 模拟面试总结文档
- 优化版答辩稿
- 学习笔记：`report_day28.md`




# ===== AI_INFRA_WEEK5 =====

# AI Infra 学习计划 - 第5周

## **Day 29 - 深入学习 CUDA Stream 与异步执行**

### 🎯 学习目标
- 理解 CUDA Stream 概念
- 掌握异步 kernel 执行与数据拷贝
- 对比同步 vs 异步性能差异

### 📌 学习重点
- CUDA Stream API（cudaStreamCreate / cudaMemcpyAsync）
- 重叠计算与数据传输
- 异步执行调试方法

### ⏱ 时间安排
- 3h：CUDA Stream 理论学习
- 3h：编写异步向量加法程序
- 2h：性能对比实验
- 2h：LeetCode（数组 & 前缀和）

### 🖥️ 代码示例
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c);
cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream);
```

### 📂 交付成果
- `vector_add_stream.cu`
- 性能对比结果
- 学习笔记：`report_day29.md`

---

## **Day 30 - CUDA 多流并发与事件**

### 🎯 学习目标
- 学习 CUDA Event 用法
- 掌握多流并发执行技巧
- 使用事件计时分析性能

### 📌 学习重点
- CUDA Event API
- 多流任务并行调度
- Event 时间测量

### ⏱ 时间安排
- 3h：CUDA Event 理论学习
- 3h：实现多流矩阵加法
- 2h：性能对比
- 2h：LeetCode（二分查找）

### 📂 交付成果
- `matrix_add_multi_stream.cu`
- 实验报告
- 学习笔记：`report_day30.md`

---

## **Day 31 - NCCL 与多GPU通信入门**

### 🎯 学习目标
- 理解 NCCL 通信框架
- 掌握多 GPU 训练/推理的基本通信原理
- 运行 NCCL AllReduce 示例

### 📌 学习重点
- NCCL API 基础
- AllReduce 操作
- 多 GPU 通信模式

### ⏱ 时间安排
- 4h：NCCL 理论学习
- 3h：运行 NCCL 示例代码
- 1h：实验报告总结
- 2h：LeetCode（图论 DFS/BFS）

### 📂 交付成果
- `nccl_allreduce.cu`
- 实验笔记
- 学习笔记：`report_day31.md`

---

## **Day 32 - NCCL 与分布式推理**

### 🎯 学习目标
- 学习 NCCL 在分布式推理中的应用
- 掌握数据并行与模型并行的概念
- 实现多 GPU 并行推理 Demo

### 📌 学习重点
- 数据并行 vs 模型并行
- NCCL 通信优化
- PyTorch 分布式推理

### ⏱ 时间安排
- 3h：学习分布式推理原理
- 3h：实现多 GPU 推理 Demo
- 2h：性能测试
- 2h：LeetCode（贪心）

### 📂 交付成果
- `multi_gpu_infer.py`
- 实验报告
- 学习笔记：`report_day32.md`

---

## **Day 33 - TensorRT 动态形状支持**

### 🎯 学习目标
- 学习 TensorRT 动态 shape 支持
- 掌握 profile 配置方法
- 测试不同输入大小下的推理性能

### 📌 学习重点
- TensorRT Dynamic Shape 配置
- Optimization Profile
- 性能对比实验

### ⏱ 时间安排
- 3h：学习 TensorRT 动态 shape 文档
- 3h：配置 profile 并运行实验
- 2h：性能对比
- 2h：LeetCode（动态规划）

### 📂 交付成果
- `tensorrt_dynamic.py`
- profile 配置文件
- 学习笔记：`report_day33.md`

---

## **Day 34 - TensorRT 多模型并行**

### 🎯 学习目标
- 掌握 TensorRT 多模型并行加载
- 实现模型 pipeline
- 对比单模型 vs 多模型性能

### 📌 学习重点
- TensorRT Engine 并行执行
- Stream 与 Engine 协作
- Pipeline 架构设计

### ⏱ 时间安排
- 4h：学习多模型并行原理
- 3h：编写多模型并行推理代码
- 1h：性能分析
- 2h：LeetCode（滑动窗口）

### 📂 交付成果
- `tensorrt_multi_model.py`
- 实验报告
- 学习笔记：`report_day34.md`

---

## **Day 35 - 第五周总结**

### 🎯 学习目标
- 总结 CUDA 与 TensorRT 高级特性
- 整理分布式与多 GPU 实验成果
- 撰写第五周学习总结

### 📌 学习重点
- CUDA Stream、Event
- NCCL 通信
- TensorRT 动态 shape 与多模型支持

### ⏱ 时间安排
- 3h：整理代码与实验报告
- 3h：绘制总结架构图
- 2h：撰写周总结
- 2h：LeetCode（综合题）

### 📂 交付成果
- `week5_summary.md`
- 架构图（多 GPU + TensorRT）
- 学习笔记：`report_day35.md`




# ===== AI_INFRA_WEEK6 =====

# AI Infra 学习计划 - 第6周

## **Day 36 - Kubernetes 高级调度策略**

### 🎯 学习目标
- 理解 Kubernetes 调度原理
- 学习 GPU 节点调度与亲和性/反亲和性
- 实现 GPU 节点优先调度 Demo

### 📌 学习重点
- K8s 调度器工作机制
- NodeSelector / NodeAffinity
- Taints 与 Tolerations

### ⏱ 时间安排
- 3h：学习调度策略
- 3h：编写 GPU 调度 YAML
- 2h：实验与验证
- 2h：LeetCode（数组 & 贪心）

### 📂 交付成果
- `gpu-scheduler-demo.yaml`
- 学习笔记：`report_day36.md`

---

## **Day 37 - Kubernetes Operator 进阶**

### 🎯 学习目标
- 学习 Operator 高级功能
- 使用 Operator 管理自定义资源生命周期
- 实现自动扩容/缩容逻辑

### 📌 学习重点
- Operator 控制循环
- 资源事件监听
- 自动化管理

### ⏱ 时间安排
- 4h：Operator 文档学习
- 3h：编写扩缩容 Operator
- 1h：实验验证
- 2h：LeetCode（二分查找）

### 📂 交付成果
- `autoscale-operator/`
- 学习笔记：`report_day37.md`

---

## **Day 38 - Service Mesh 入门（Istio）**

### 🎯 学习目标
- 理解 Service Mesh 概念
- 学习 Istio 架构
- 在 K8s 中部署 Istio

### 📌 学习重点
- Sidecar 模式
- 流量治理
- 可观测性增强

### ⏱ 时间安排
- 3h：Service Mesh 理论学习
- 3h：Istio 部署
- 2h：实验验证（流量分流）
- 2h：LeetCode（图论 BFS/DFS）

### 📂 交付成果
- `istio-demo.yaml`
- 学习笔记：`report_day38.md`

---

## **Day 39 - Istio 高级流量治理**

### 🎯 学习目标
- 掌握 Istio 高级流量控制
- 实现 Canary 发布
- 使用 Istio Dashboard 观察流量

### 📌 学习重点
- VirtualService / DestinationRule
- Canary 部署策略
- 可观测性

### ⏱ 时间安排
- 3h：学习流量治理策略
- 3h：实现 Canary 部署
- 2h：可观测性实验
- 2h：LeetCode（字符串处理）

### 📂 交付成果
- `canary-deploy.yaml`
- 学习笔记：`report_day39.md`

---

## **Day 40 - CI/CD 基础（Jenkins/GitLab CI）**

### 🎯 学习目标
- 学习 CI/CD 基本流程
- 使用 Jenkins 或 GitLab CI 部署 AI 模型服务
- 自动化构建 Docker 镜像

### 📌 学习重点
- CI/CD pipeline
- 自动化测试
- 自动部署

### ⏱ 时间安排
- 3h：CI/CD 理论学习
- 3h：编写 Jenkinsfile / GitLab CI 配置
- 2h：运行实验
- 2h：LeetCode（动态规划）

### 📂 交付成果
- `Jenkinsfile` / `.gitlab-ci.yml`
- 学习笔记：`report_day40.md`

---

## **Day 41 - CI/CD 高级（Helm + ArgoCD）**

### 🎯 学习目标
- 学习 Helm Chart 打包
- 使用 ArgoCD 实现 GitOps 部署
- 自动化管理 AI 推理平台

### 📌 学习重点
- Helm Chart 语法
- ArgoCD 部署流程
- GitOps 实践

### ⏱ 时间安排
- 3h：学习 Helm Chart
- 3h：使用 ArgoCD 部署
- 2h：实验验证
- 2h：LeetCode（滑动窗口）

### 📂 交付成果
- `ai-infra-chart/`
- ArgoCD 部署配置
- 学习笔记：`report_day41.md`

---

## **Day 42 - 第六周总结**

### 🎯 学习目标
- 总结 Kubernetes、Service Mesh、CI/CD 学习成果
- 整合到 AI 推理平台项目
- 更新简历项目亮点

### 📌 学习重点
- GPU 调度
- Istio Canary 部署
- GitOps 工作流

### ⏱ 时间安排
- 3h：整理代码与文档
- 3h：更新项目架构图
- 2h：撰写周总结
- 2h：LeetCode（综合题）

### 📂 交付成果
- `week6_summary.md`
- 完整项目架构图（K8s + Istio + ArgoCD）
- 学习笔记：`report_day42.md`




# ===== AI_INFRA_WEEK7 =====

# AI Infra 学习计划 - 第7周

## **Day 43 - 分布式训练与推理概念**

### 🎯 学习目标
- 理解分布式训练与推理的区别
- 学习数据并行与模型并行原理
- 掌握常见分布式框架（Horovod、DeepSpeed）

### 📌 学习重点
- 数据并行 vs 模型并行
- 参数同步机制
- 分布式推理场景

### ⏱ 时间安排
- 3h：学习分布式训练与推理原理
- 3h：阅读 Horovod 文档并运行示例
- 2h：调试分布式训练脚本
- 2h：LeetCode（数组 & 动态规划）

### 📂 交付成果
- Horovod 示例脚本
- 学习笔记：`report_day43.md`

---

## **Day 44 - NCCL 通信与多机多卡**

### 🎯 学习目标
- 学习 NCCL 在多机多卡中的应用
- 掌握 AllReduce、Broadcast、ReduceScatter
- 实现多机通信实验

### 📌 学习重点
- NCCL 通信模式
- 多机分布式训练
- 通信优化技巧

### ⏱ 时间安排
- 4h：学习 NCCL 多机通信
- 3h：运行分布式通信实验
- 1h：撰写实验报告
- 2h：LeetCode（二分 & 贪心）

### 📂 交付成果
- `nccl_multi_node.py`
- 实验报告
- 学习笔记：`report_day44.md`

---

## **Day 45 - DeepSpeed 推理优化**

### 🎯 学习目标
- 学习 DeepSpeed Inference 特性
- 使用 DeepSpeed 部署大模型推理
- 对比 Triton 与 DeepSpeed 性能

### 📌 学习重点
- DeepSpeed ZeRO-Inference
- 分布式推理优化
- 与 Triton 集成

### ⏱ 时间安排
- 3h：DeepSpeed 文档学习
- 3h：部署一个 BERT 推理服务
- 2h：性能对比实验
- 2h：LeetCode（链表 & 双指针）

### 📂 交付成果
- `deepspeed_infer.py`
- 实验报告
- 学习笔记：`report_day45.md`

---

## **Day 46 - Ray Serve 入门**

### 🎯 学习目标
- 学习 Ray Serve 框架
- 使用 Ray Serve 部署简单推理服务
- 理解 Ray 在分布式推理中的优势

### 📌 学习重点
- Ray Serve 基础 API
- 部署多模型服务
- Ray 与 K8s 集成

### ⏱ 时间安排
- 3h：Ray Serve 文档学习
- 3h：编写并运行推理服务
- 2h：测试并验证性能
- 2h：LeetCode（字符串处理）

### 📂 交付成果
- `ray_serve_demo.py`
- 学习笔记：`report_day46.md`

---

## **Day 47 - Ray Serve 高级功能**

### 🎯 学习目标
- 学习 Ray Serve 路由与负载均衡
- 部署多模型推理服务
- 实现简单的 A/B 测试

### 📌 学习重点
- Ray Serve 路由规则
- 动态扩缩容
- 多模型部署

### ⏱ 时间安排
- 4h：Ray Serve 高级特性学习
- 3h：编写多模型部署代码
- 1h：测试与验证
- 2h：LeetCode（图论 DFS/BFS）

### 📂 交付成果
- `ray_serve_multi.py`
- 学习笔记：`report_day47.md`

---

## **Day 48 - 分布式推理平台对比**

### 🎯 学习目标
- 对比 Triton、DeepSpeed、Ray Serve 在分布式推理中的应用
- 从性能、易用性、可扩展性进行分析
- 撰写技术调研报告

### 📌 学习重点
- Triton 优势与局限
- DeepSpeed 优势与局限
- Ray Serve 优势与局限

### ⏱ 时间安排
- 4h：整理实验结果
- 4h：撰写技术调研报告
- 2h：LeetCode（综合题）

### 📂 交付成果
- `distributed_inference_comparison.md`
- 学习笔记：`report_day48.md`

---

## **Day 49 - 第七周总结**

### 🎯 学习目标
- 总结分布式推理与多机多卡实验成果
- 更新 AI 推理平台项目，增加分布式模块
- 完善简历描述

### 📌 学习重点
- 分布式推理框架对比
- 项目扩展
- 简历亮点提炼

### ⏱ 时间安排
- 3h：整理代码与文档
- 3h：绘制分布式架构图
- 2h：撰写周总结
- 2h：LeetCode（综合题）

### 📂 交付成果
- `week7_summary.md`
- 分布式推理架构图
- 学习笔记：`report_day49.md`




# ===== AI_INFRA_WEEK8 =====

# AI Infra 学习计划 - 第8周

## **Day 50 - 模型压缩与量化基础**

### 🎯 学习目标
- 理解模型压缩的意义与方法
- 学习常见量化方法（Post-training Quantization, QAT）
- 在 PyTorch 中实现简单的量化实验

### 📌 学习重点
- 量化原理
- PyTorch 量化 API
- 精度与性能对比

### ⏱ 时间安排
- 3h：学习模型压缩与量化原理
- 3h：PyTorch 量化实验
- 2h：性能与精度对比
- 2h：LeetCode（数组 & 哈希）

### 📂 交付成果
- `quantization_demo.py`
- 实验报告
- 学习笔记：`report_day50.md`

---

## **Day 51 - 知识蒸馏入门**

### 🎯 学习目标
- 学习知识蒸馏原理
- 使用大模型作为 teacher，小模型作为 student
- 对比精度与性能差异

### 📌 学习重点
- Distillation Loss
- Teacher-Student 架构
- PyTorch 蒸馏实现

### ⏱ 时间安排
- 3h：知识蒸馏理论学习
- 3h：实现蒸馏实验（MNIST/CIFAR10）
- 2h：精度对比
- 2h：LeetCode（字符串）

### 📂 交付成果
- `distillation_demo.py`
- 实验报告
- 学习笔记：`report_day51.md`

---

## **Day 52 - 模型裁剪与稀疏化**

### 🎯 学习目标
- 学习模型裁剪（Pruning）方法
- 理解权重稀疏化的作用
- 使用 PyTorch 实现模型裁剪实验

### 📌 学习重点
- 剪枝原理
- 稀疏矩阵计算
- PyTorch 剪枝 API

### ⏱ 时间安排
- 3h：学习模型剪枝原理
- 3h：实现剪枝实验
- 2h：稀疏化性能测试
- 2h：LeetCode（动态规划）

### 📂 交付成果
- `pruning_demo.py`
- 实验报告
- 学习笔记：`report_day52.md`

---

## **Day 53 - 模型加速工具链对比**

### 🎯 学习目标
- 对比 TensorRT、ONNX Runtime、OpenVINO 在推理加速中的应用
- 从性能、易用性、兼容性分析
- 撰写技术调研报告

### 📌 学习重点
- 各推理框架的优势与局限
- GPU vs CPU 加速对比
- 工业界使用场景

### ⏱ 时间安排
- 3h：研究推理框架
- 3h：运行实验对比
- 2h：撰写调研报告
- 2h：LeetCode（贪心）

### 📂 交付成果
- `inference_framework_comparison.md`
- 学习笔记：`report_day53.md`

---

## **Day 54 - GPU Profiling 工具使用**

### 🎯 学习目标
- 学习 Nsight Systems、Nsight Compute 的使用方法
- 分析 CUDA Kernel 性能
- 找出性能瓶颈

### 📌 学习重点
- Nsight 工具使用
- Kernel Profiling
- 性能优化

### ⏱ 时间安排
- 4h：学习 Nsight 工具
- 3h：运行 profiling 实验
- 1h：撰写实验报告
- 2h：LeetCode（滑动窗口）

### 📂 交付成果
- Nsight 报告截图
- 学习笔记：`report_day54.md`

---

## **Day 55 - AI 推理平台优化**

### 🎯 学习目标
- 将模型压缩、量化、蒸馏成果应用到 AI 推理平台
- 优化推理性能
- 更新架构设计

### 📌 学习重点
- 模型优化与平台结合
- 性能对比实验
- 架构更新

### ⏱ 时间安排
- 4h：应用优化成果
- 3h：测试平台性能
- 1h：更新架构图
- 2h：LeetCode（二分查找）

### 📂 交付成果
- 更新版推理平台代码
- 架构图（优化后）
- 学习笔记：`report_day55.md`

---

## **Day 56 - 第八周总结**

### 🎯 学习目标
- 总结模型优化相关学习成果
- 更新项目文档
- 完善简历亮点

### 📌 学习重点
- 模型压缩、量化、蒸馏
- 推理框架对比
- Profiling 优化

### ⏱ 时间安排
- 3h：整理代码与实验结果
- 3h：撰写周总结
- 2h：更新简历描述
- 2h：LeetCode（综合题）

### 📂 交付成果
- `week8_summary.md`
- 完整实验报告
- 学习笔记：`report_day56.md`




# ===== AI_INFRA_WEEK9 =====

# AI Infra 学习计划 - 第9周

## **Day 57 - 面试准备：操作系统与并发**

### 🎯 学习目标
- 复习操作系统常考知识点
- 掌握并发编程基础
- 准备相关面试题

### 📌 学习重点
- 进程 vs 线程
- 锁、信号量、条件变量
- 死锁与解决方案

### ⏱ 时间安排
- 3h：复习操作系统并发概念
- 3h：实现多线程生产者-消费者模型
- 2h：整理面试题
- 2h：LeetCode（并发题 / 多线程题）

### 📂 交付成果
- `producer_consumer.cpp`
- `interview_os_concurrency.md`
- 学习笔记：`report_day57.md`

---

## **Day 58 - 面试准备：计算机网络**

### 🎯 学习目标
- 复习计算机网络常考知识点
- 掌握 TCP/IP、UDP 区别与应用
- 准备相关面试题

### 📌 学习重点
- TCP 三次握手、四次挥手
- 拥塞控制
- gRPC 原理

### ⏱ 时间安排
- 3h：复习网络协议
- 3h：编写简单 TCP/UDP Echo Server
- 2h：整理面试题
- 2h：LeetCode（字符串 & 哈希）

### 📂 交付成果
- `tcp_udp_server.cpp`
- `interview_network.md`
- 学习笔记：`report_day58.md`

---

## **Day 59 - 面试准备：数据库与存储**

### 🎯 学习目标
- 复习数据库相关知识点
- 理解分布式存储系统架构
- 准备相关面试题

### 📌 学习重点
- MySQL 索引与事务
- Redis 缓存
- 分布式存储（HDFS、Ceph）

### ⏱ 时间安排
- 3h：复习数据库基础
- 3h：写 Redis 缓存 Demo
- 2h：整理面试题
- 2h：LeetCode（二叉树）

### 📂 交付成果
- `redis_demo.py`
- `interview_database.md`
- 学习笔记：`report_day59.md`

---

## **Day 60 - 面试准备：系统设计**

### 🎯 学习目标
- 学习系统设计方法论
- 设计 AI 推理平台架构
- 准备相关面试题

### 📌 学习重点
- 高并发系统设计
- 弹性伸缩与容灾
- AI 推理服务架构设计

### ⏱ 时间安排
- 4h：学习系统设计案例
- 3h：绘制推理平台架构图
- 1h：整理系统设计问答
- 2h：LeetCode（图论 BFS/DFS）

### 📂 交付成果
- `ai_inference_system_design.md`
- 架构图（系统设计版）
- 学习笔记：`report_day60.md`

---

## **Day 61 - 项目答辩准备（技术难点）**

### 🎯 学习目标
- 整理项目难点与解决方案
- 准备技术深挖回答
- 撰写答辩稿

### 📌 学习重点
- CUDA 优化难点
- TensorRT 插件开发
- K8s 调度与监控体系

### ⏱ 时间安排
- 3h：总结项目技术难点
- 3h：撰写详细答辩稿
- 2h：模拟问答
- 2h：LeetCode（动态规划）

### 📂 交付成果
- `project_qa.md`
- 答辩文档
- 学习笔记：`report_day61.md`

---

## **Day 62 - 项目答辩准备（业务价值）**

### 🎯 学习目标
- 提炼项目亮点与业务价值
- 总结对企业的价值（性能提升、成本优化）
- 准备 STAR 法则回答

### 📌 学习重点
- 项目业务价值
- 面试故事化表达
- STAR 法则练习

### ⏱ 时间安排
- 3h：撰写项目价值总结
- 3h：准备 STAR 回答
- 2h：模拟答辩演讲
- 2h：LeetCode（数组 & 贪心）

### 📂 交付成果
- `project_business_value.md`
- STAR 回答文档
- 学习笔记：`report_day62.md`

---

## **Day 63 - 第九周总结与模拟面试**

### 🎯 学习目标
- 进行一次完整模拟面试
- 包含算法、系统设计、项目讲解
- 复盘面试表现

### 📌 学习重点
- 算法手写
- 系统设计答辩
- 项目价值讲述

### ⏱ 时间安排
- 3h：模拟算法题
- 3h：模拟系统设计
- 2h：模拟项目答辩
- 2h：复盘总结

### 📂 交付成果
- 模拟面试总结文档
- 学习笔记：`report_day63.md`




# ===== AI_INFRA_WEEK10 =====

# AI Infra 学习计划 - 第10周

## **Day 64 - 高频算法题专项（数组与哈希表）**

### 🎯 学习目标
- 强化 LeetCode 高频题
- 掌握数组与哈希表常见解题思路
- 总结常见面试套路

### 📌 学习重点
- Two Sum, Three Sum
- LRU Cache
- 子数组问题（和为 K 的子数组）

### ⏱ 时间安排
- 5h：刷题（10 道数组/哈希表）
- 3h：总结解题模板
- 2h：编写题解文档

### 📂 交付成果
- `leetcode_array_hash.md`
- 学习笔记：`report_day64.md`

---

## **Day 65 - 高频算法题专项（链表与堆栈）**

### 🎯 学习目标
- 掌握链表与栈/队列题型
- 理解指针操作
- 熟悉堆栈模拟场景题

### 📌 学习重点
- 反转链表
- 合并 K 个有序链表
- 单调栈/队列应用

### ⏱ 时间安排
- 5h：刷题（8 道链表/栈队列）
- 3h：总结常见技巧
- 2h：编写题解文档

### 📂 交付成果
- `leetcode_list_stack.md`
- 学习笔记：`report_day65.md`

---

## **Day 66 - 高频算法题专项（二分与动态规划）**

### 🎯 学习目标
- 熟练掌握二分查找及变体
- 掌握动态规划经典题
- 总结状态转移方程思路

### 📌 学习重点
- 二分查找、旋转数组最小值
- 背包问题
- 最长子序列问题

### ⏱ 时间安排
- 5h：刷题（10 道二分/DP）
- 3h：总结题解与模板
- 2h：编写题解文档

### 📂 交付成果
- `leetcode_binary_dp.md`
- 学习笔记：`report_day66.md`

---

## **Day 67 - 高频算法题专项（图论与搜索）**

### 🎯 学习目标
- 熟悉图论与搜索题型
- DFS/BFS 模板总结
- 掌握并查集、最短路径

### 📌 学习重点
- DFS/BFS 遍历
- Dijkstra、Floyd 算法
- 并查集应用

### ⏱ 时间安排
- 5h：刷题（8 道图论/搜索）
- 3h：整理常用算法模板
- 2h：编写题解文档

### 📂 交付成果
- `leetcode_graph.md`
- 学习笔记：`report_day67.md`

---

## **Day 68 - 系统设计题专项**

### 🎯 学习目标
- 学习系统设计题思路
- 掌握常见组件设计（缓存、消息队列、负载均衡）
- 练习 AI 推理服务系统设计题

### 📌 学习重点
- 高并发系统设计
- 缓存一致性
- 负载均衡策略

### ⏱ 时间安排
- 4h：学习系统设计案例
- 4h：设计 AI 推理服务架构
- 2h：编写系统设计答辩稿

### 📂 交付成果
- `system_design_ai_infer.md`
- 学习笔记：`report_day68.md`

---

## **Day 69 - 项目深挖问答准备**

### 🎯 学习目标
- 准备项目中可能被深挖的问题
- 总结项目架构优化点
- 模拟面试问答

### 📌 学习重点
- CUDA 优化细节
- TensorRT 动态 shape、Plugin
- K8s 调度与 HPA

### ⏱ 时间安排
- 4h：整理深挖问题
- 4h：模拟问答演练
- 2h：编写 Q&A 文档

### 📂 交付成果
- `project_deep_dive.md`
- 学习笔记：`report_day69.md`

---

## **Day 70 - 第十周总结与模拟面试**

### 🎯 学习目标
- 总结高频题与系统设计
- 进行完整模拟面试
- 复盘改进答题策略

### 📌 学习重点
- 高频算法题
- 系统设计
- 项目答辩

### ⏱ 时间安排
- 3h：模拟算法题
- 3h：系统设计模拟
- 2h：项目答辩模拟
- 2h：复盘总结

### 📂 交付成果
- 模拟面试总结文档
- 学习笔记：`report_day70.md`




# ===== AI_INFRA_WEEK11 =====

# AI Infra 学习计划 - 第11周

## **Day 71 - 系统设计专项（缓存与一致性）**

### 🎯 学习目标
- 学习缓存一致性问题
- 掌握常见缓存策略
- 准备相关面试题

### 📌 学习重点
- Cache Aside, Read Through, Write Back
- 缓存雪崩、击穿、穿透
- Redis 与分布式缓存

### ⏱ 时间安排
- 4h：学习缓存一致性案例
- 3h：编写缓存 Demo
- 1h：整理面试题
- 2h：LeetCode（哈希）

### 📂 交付成果
- `cache_demo.py`
- `interview_cache.md`
- 学习笔记：`report_day71.md`

---

## **Day 72 - 系统设计专项（消息队列与异步架构）**

### 🎯 学习目标
- 理解消息队列在系统设计中的作用
- 掌握 Kafka/RabbitMQ 应用场景
- 准备相关面试题

### 📌 学习重点
- MQ 的作用（解耦、削峰、异步）
- Kafka 主题与分区
- 消费者组与偏移量

### ⏱ 时间安排
- 4h：学习 MQ 架构原理
- 3h：编写 Kafka 消费者/生产者 Demo
- 1h：整理面试题
- 2h：LeetCode（队列）

### 📂 交付成果
- `kafka_demo.py`
- `interview_mq.md`
- 学习笔记：`report_day72.md`

---

## **Day 73 - 系统设计专项（负载均衡与网关）**

### 🎯 学习目标
- 学习负载均衡常见策略
- 掌握 API Gateway 在推理平台中的作用
- 准备相关面试题

### 📌 学习重点
- 负载均衡算法（轮询、哈希、一致性哈希）
- API Gateway 功能
- Nginx / Envoy 使用

### ⏱ 时间安排
- 4h：学习负载均衡原理
- 3h：部署 Nginx 反向代理
- 1h：整理面试题
- 2h：LeetCode（堆）

### 📂 交付成果
- `nginx_gateway.conf`
- `interview_load_balancer.md`
- 学习笔记：`report_day73.md`

---

## **Day 74 - DevOps 与运维体系**

### 🎯 学习目标
- 学习 DevOps 思想
- 掌握监控、日志、报警体系
- 理解 SRE 在 AI Infra 中的职责

### 📌 学习重点
- CI/CD 回顾
- 日志采集（ELK/EFK）
- SLA/SLO/SLI

### ⏱ 时间安排
- 4h：学习 DevOps/SRE 理论
- 3h：配置 ELK Stack
- 1h：整理面试题
- 2h：LeetCode（字符串）

### 📂 交付成果
- ELK 配置文件
- `interview_devops.md`
- 学习笔记：`report_day74.md`

---

## **Day 75 - 高频面试题模拟（技术综合）**

### 🎯 学习目标
- 模拟真实面试场景
- 综合考察操作系统、网络、数据库、系统设计
- 记录答题表现

### 📌 学习重点
- 操作系统并发
- 网络协议
- 数据库索引与事务
- 系统设计综合题

### ⏱ 时间安排
- 4h：模拟技术面试
- 3h：整理问题与答案
- 1h：复盘答题不足
- 2h：LeetCode（动态规划）

### 📂 交付成果
- `mock_interview_day75.md`
- 学习笔记：`report_day75.md`

---

## **Day 76 - 高频面试题模拟（AI Infra 技术栈）**

### 🎯 学习目标
- 模拟 AI Infra 专项面试
- 包括 CUDA、TensorRT、Triton、K8s、监控
- 总结面试亮点

### 📌 学习重点
- CUDA 优化与内存模型
- TensorRT Engine 与 Plugin
- Triton 部署与监控

### ⏱ 时间安排
- 4h：模拟 AI Infra 技术面试
- 3h：记录与复盘
- 1h：优化答题稿
- 2h：LeetCode（图论）

### 📂 交付成果
- `mock_interview_day76.md`
- 学习笔记：`report_day76.md`

---

## **Day 77 - 第十一周总结**

### 🎯 学习目标
- 总结系统设计与面试专项学习成果
- 更新 AI 推理平台简历描述
- 完善面试准备材料

### 📌 学习重点
- 系统设计专项（缓存、MQ、负载均衡）
- DevOps 与 SRE
- 面试答题优化

### ⏱ 时间安排
- 3h：整理文档
- 3h：更新简历与项目总结
- 2h：编写周总结
- 2h：LeetCode（综合题）

### 📂 交付成果
- `week11_summary.md`
- 完整简历更新版
- 学习笔记：`report_day77.md`




# ===== AI_INFRA_WEEK12 =====

# AI Infra 学习计划 - 第12周

## **Day 78 - 高频面试题整理（操作系统 & 网络）**

### 🎯 学习目标
- 总结操作系统与网络高频面试题
- 编写答题模板
- 模拟答题

### 📌 学习重点
- 操作系统：进程/线程、内存管理、锁
- 网络：TCP 三次握手、四次挥手、HTTP/2 与 gRPC
- 高频面试考点总结

### ⏱ 时间安排
- 4h：学习与整理题库
- 3h：编写答题文档
- 1h：模拟答题演练
- 2h：LeetCode（多线程题）

### 📂 交付成果
- `interview_os_network.md`
- 学习笔记：`report_day78.md`

---

## **Day 79 - 高频面试题整理（数据库 & 分布式系统）**

### 🎯 学习目标
- 总结数据库与分布式系统高频面试题
- 编写答题模板
- 模拟答题

### 📌 学习重点
- 数据库：事务、索引、锁机制
- 分布式系统：CAP 定理、共识算法、分布式锁
- 高频面试考点总结

### ⏱ 时间安排
- 4h：学习与整理题库
- 3h：编写答题文档
- 1h：模拟答题演练
- 2h：LeetCode（SQL/DB 题）

### 📂 交付成果
- `interview_db_distributed.md`
- 学习笔记：`report_day79.md`

---

## **Day 80 - 高频面试题整理（AI Infra 专项）**

### 🎯 学习目标
- 总结 AI Infra 专项高频题
- 形成答题模板
- 模拟答题

### 📌 学习重点
- CUDA 内存优化
- TensorRT Engine 构建与 Plugin
- Triton + K8s 部署实践
- Prometheus + Grafana 监控实践

### ⏱ 时间安排
- 4h：学习与整理题库
- 3h：编写答题文档
- 1h：模拟答题演练
- 2h：LeetCode（综合题）

### 📂 交付成果
- `interview_ai_infra.md`
- 学习笔记：`report_day80.md`

---

## **Day 81 - 模拟面试（算法 + 系统设计）**

### 🎯 学习目标
- 模拟真实技术面试（算法 + 系统设计）
- 训练答题思维
- 发现不足并改进

### 📌 学习重点
- LeetCode 高频题
- 系统设计题（AI 推理平台、分布式缓存）
- 答题条理性

### ⏱ 时间安排
- 3h：模拟算法题
- 4h：模拟系统设计题
- 3h：复盘与改进

### 📂 交付成果
- `mock_interview_day81.md`
- 学习笔记：`report_day81.md`

---

## **Day 82 - 模拟面试（AI Infra 技术栈）**

### 🎯 学习目标
- 模拟 AI Infra 技术专项面试
- CUDA、TensorRT、Triton、K8s
- 总结改进方向

### 📌 学习重点
- CUDA 优化
- TensorRT 动态 shape
- Triton Inference Server
- K8s HPA 与调度策略

### ⏱ 时间安排
- 4h：模拟专项面试
- 3h：复盘与改进
- 3h：总结答题稿

### 📂 交付成果
- `mock_interview_day82.md`
- 学习笔记：`report_day82.md`

---

## **Day 83 - 模拟全流程面试（综合）**

### 🎯 学习目标
- 完整模拟面试流程
- 包括算法、系统设计、AI Infra 技术、项目答辩
- 提升面试表现

### 📌 学习重点
- 算法题 2 道
- 系统设计题 1 道
- 项目答辩 10 分钟

### ⏱ 时间安排
- 3h：模拟完整面试
- 3h：复盘与反馈
- 2h：优化答题稿
- 2h：LeetCode（复习）

### 📂 交付成果
- `mock_full_interview_day83.md`
- 学习笔记：`report_day83.md`

---

## **Day 84 - 第十二周总结与最终准备**

### 🎯 学习目标
- 总结 12 周学习成果
- 整理所有文档
- 准备最终面试材料

### 📌 学习重点
- 面试题库
- 系统设计稿
- 项目总结与亮点

### ⏱ 时间安排
- 3h：整理面试材料
- 3h：更新简历与项目亮点
- 2h：编写周总结
- 2h：自由复习

### 📂 交付成果
- `week12_summary.md`
- 最终简历 + 项目总结版
- 学习笔记：`report_day84.md`




# ===== AI_INFRA_WEEK13 =====

# AI Infra 学习计划 - 第13周 (Day85~Day90)

## **Day 85 - 项目复盘与优化（推理平台）**

### 🎯 学习目标
- 回顾 AI 推理平台的整体架构
- 分析项目亮点与不足
- 优化简历中的项目描述

### 📌 学习重点
- 系统架构回顾（Docker + K8s + Triton）
- 技术难点总结（TensorRT、CUDA 优化）
- 亮点与可讲述点提炼

### ⏱ 时间安排
- 4h：项目复盘
- 3h：整理项目亮点
- 2h：更新简历
- 1h：自由复习

### 📂 交付成果
- `project_review.md`
- 优化后的简历项目描述

---

## **Day 86 - 算法专项复习**

### 🎯 学习目标
- 回顾 LeetCode 高频题
- 分类复习常见算法
- 提升答题速度

### 📌 学习重点
- 数组 & 哈希
- 动态规划
- 图论 & 并查集

### ⏱ 时间安排
- 4h：LeetCode 高频题
- 3h：复习解题模板
- 2h：模拟笔试
- 1h：错题总结

### 📂 交付成果
- `leetcode_review_day86.md`
- 学习笔记：`report_day86.md`

---

## **Day 87 - 系统设计专项复习**

### 🎯 学习目标
- 回顾系统设计高频题
- 整理答题模板
- 模拟答题场景

### 📌 学习重点
- 缓存设计
- 消息队列
- 负载均衡
- API Gateway

### ⏱ 时间安排
- 4h：复习系统设计案例
- 3h：模拟答题
- 2h：文档总结
- 1h：自由练习

### 📂 交付成果
- `system_design_review.md`
- 学习笔记：`report_day87.md`

---

## **Day 88 - 面试综合演练（AI Infra 专项）**

### 🎯 学习目标
- 模拟 AI Infra 面试
- 包括 CUDA、TensorRT、Triton、K8s、监控
- 查漏补缺

### 📌 学习重点
- CUDA 优化与显存管理
- TensorRT Engine 构建
- Triton Inference 部署
- Prometheus + Grafana

### ⏱ 时间安排
- 4h：模拟专项面试
- 3h：复盘与改进
- 2h：整理答题稿
- 1h：自由复习

### 📂 交付成果
- `mock_infra_day88.md`
- 学习笔记：`report_day88.md`

---

## **Day 89 - 模拟全流程面试（最终彩排）**

### 🎯 学习目标
- 完整模拟 1 次真实面试流程
- 包括算法、系统设计、AI Infra 技术、项目答辩
- 全流程打磨

### 📌 学习重点
- 算法 2 道
- 系统设计题 1 道
- 项目答辩（10-15 分钟）

### ⏱ 时间安排
- 4h：模拟全流程面试
- 3h：复盘与反馈
- 2h：优化答题稿
- 1h：自由练习

### 📂 交付成果
- `mock_final_day89.md`
- 学习笔记：`report_day89.md`

---

## **Day 90 - 最终收尾与自由 buffer**

### 🎯 学习目标
- 整理全部学习成果
- 完善最终简历
- 进行灵活复习与查漏补缺

### 📌 学习重点
- 学习文档整理
- 简历优化
- 自由复盘

### ⏱ 时间安排
- 3h：整理全部笔记
- 3h：优化最终简历
- 2h：复盘项目与系统设计
- 2h：自由复习 & 放松

### 📂 交付成果
- `final_summary.md`
- 最终版简历 + 项目亮点
- 学习笔记：`report_day90.md`


