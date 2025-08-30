
# 📘 Week 1 学习计划（合并加强版）

## **Day 1 - Docker & GPU 容器化**

### 🎯 学习目标
- 掌握 Docker 基础命令与容器化流程  
- 构建 PyTorch 推理环境镜像  
- 在容器中运行 GPU 程序（nvidia-docker）

### 📌 学习重点
- 常用命令：`run / ps / exec / logs / rm`  
- Dockerfile 构建与缓存层优化  
- 卷挂载、网络、`--gpus all`  
- nvidia-container-runtime 工作方式

### ⏱ 时间安排
- 3h：基础命令练习与镜像结构认识  
- 2h：编写 Dockerfile 打包 PyTorch/ONNX 环境  
- 2h：安装并验证 nvidia-docker  
- 1h：总结 `report_day1.md`  
- 2h：LeetCode（数组 + 哈希）

### 🖥️ 命令示例
```bash
docker pull pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
docker run --gpus all -it --rm -v $PWD:/ws pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime bash
python -c "import torch;print('CUDA:', torch.cuda.is_available())"
```

### 📂 交付成果
- `Dockerfile`（含 PyTorch+ONNX+工具链）  
- 运行截图（`torch.cuda.is_available()` 为 True）  
- `report_day1.md`

### 🔍 深度追问
1. 镜像分层（UnionFS/OverlayFS）如何命中缓存？哪些命令会破坏缓存链？  
2. 多阶段构建如何避免把编译产物依赖带入 runtime 层？  
3. OCI Image/Runtime 规范与 Docker 的关系是什么？  
4. `nvidia-container-runtime` 如何将 GPU 能力注入容器（device+cgroup+libcuda.so 映射）？  
5. `--privileged` 与 `--cap-add SYS_ADMIN` 的差异与最小权限原则？  
6. 容器只读根文件系统（`--read-only`）对日志/临时目录的影响与最佳实践？  

### 🧪 实验
1. 写两版 Dockerfile：①随写`RUN apt ...`；②合理分层+清理缓存 → 比体积与 build 时间。  
2. 多阶段构建：`builder` + `runtime`，观察 runtime 层体积变化。  
3. `micromamba` vs `apt` vs `pip`：构建镜像，对比体积与冷启动时间。  
4. 只读根文件系统 + 写入卷映射，验证训练日志和临时文件可用性。  
5. 设置 `--memory=512m` 与 `--pids-limit`，在容器内触发 OOM 和进程数上限，记录行为差异。  

---

## **Day 2 - Kubernetes 基础 & GPU Pod**

### 🎯 学习目标
- 掌握 K8s 基础对象：Pod/Deployment/Service  
- 在集群中运行 GPU Pod  
- 使用 Service 暴露推理服务

### 📌 学习重点
- `kubectl` 常用子命令与资源查看  
- nvidia/k8s-device-plugin 安装  
- Service（ClusterIP/NodePort）与端口映射

### ⏱ 时间安排
- 3h：安装 Minikube/Kind & 基本命令  
- 3h：部署 GPU Pod（含 device-plugin）  
- 2h：通过 Service 暴露服务  
- 2h：LeetCode（栈 + 队列）

### 🖥️ 命令示例
```bash
minikube start --driver=docker --gpus=all
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
kubectl create deployment gpu-test --image=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
kubectl expose deployment gpu-test --type=NodePort --port=8888
kubectl get pods -o wide
```

### 📂 交付成果
- `k8s-gpu-demo.yaml`  
- `kubectl get pods`/`describe` 截图  
- `report_day2.md`

### 🔍 深度追问
1. Device Plugin 如何向 kubelet 上报 `resource=nvidia.com/gpu`？调度器如何据此筛选节点？  
2. `requests/limits` 如何影响 QoS 等级（Guaranteed/Burstable/BestEffort）与驱逐策略？  
3. `kube-proxy` 的 `iptables` 与 `ipvs` 模式在转发性能和一致性上的差异？  
4. Node Feature Discovery 如何暴露 GPU/NUMA/PCIe 拓扑信息？  
5. `RuntimeClass` 与 `containerd`/`crio` 的关联能带来哪些隔离/性能差异？  
6. `PodDisruptionBudget`、`PriorityClass` 在在线推理稳定性中的作用？  

### 🧪 实验
1. 安装 nvidia device plugin 后，创建 `requests: {"nvidia.com/gpu": 1}` Pod→观察调度事件。  
2. 制造 OOM：①`requests=1Gi, limits=512Mi`；②`requests=512Mi, limits=1Gi` → 看谁更易被驱逐。  
3. 同一服务用 `port-forward / NodePort / Ingress` 暴露，压测比较时延与吞吐。  
4. 给 GPU 节点打 `taint`，给 Pod 加 `tolerations`，验证“只在 GPU 节点调度”。  
5. 配置 `PDB minAvailable=1`，滚动更新 Deployment，观察是否阻止无序中断。  

---

## **Day 3 - PyTorch → ONNX**

### 🎯 学习目标
- 训练/保存一个简单 PyTorch 模型（MNIST/CIFAR10）  
- 导出为 ONNX  
- 使用 ONNX Runtime 推理

### 📌 学习重点
- 模型定义/保存/加载  
- ONNX 导出参数（`opset_version`、动态维度）  
- ONNX Runtime InferenceSession

### ⏱ 时间安排
- 4h：训练与保存模型  
- 2h：导出 ONNX 与本地推理  
- 2h：容器/脚本化运行  
- 2h：LeetCode（链表 + 二分）

### 🖥️ 代码示例
```python
import torch, torch.nn as nn
m = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10))
x = torch.randn(1, 1, 28, 28)
torch.onnx.export(m, x, "mlp.onnx", input_names=["input"], output_names=["logits"], opset_version=17)
```

### 📂 交付成果
- `mlp.onnx`  
- `onnx_infer.py`（ORT 推理脚本）  
- `report_day3.md`

### 🔍 深度追问
1. ONNX 的算子标准化如何保证跨框架一致性？哪个环节最易出现 break？  
2. `dynamic_axes` 导出后，推理端 shape 推断与优化有何变化？  
3. ORT 的图优化级别（`disable_basic/extended/all`）分别做了什么？  
4. ORT 的 Execution Provider 如何选择内核？  
5. 哪些 PyTorch 模式导出最易失败（控制流/自定义 op）？  
6. ONNX 模型 initializers/constants 对加载时间/体积影响？  

### 🧪 实验
1. 导出 opset=13/17 两版，跑 ORT `opt_level=basic/extended/all`，记录延迟差。  
2. 动态 batch（1/8/32）输入测试 ORT CUDA EP 的时延/吞吐。  
3. 关闭 ORT 所有优化 vs 默认全开，比较性能与内存峰值。  
4. 替换一层 ReLU 为自定义函数，观察导出/运行报错情况。  
5. 用 Netron 查看 ONNX 图，对比常量折叠前后节点数量。  

---

## **Day 4 - TensorRT 优化**

### 🎯 学习目标
- 用 `trtexec`/API 将 ONNX 转换为 TensorRT engine  
- 测试 FP16/INT8 加速效果

### 📌 学习重点
- Builder/Network/Config 基础  
- FP16/INT8 校准要点  
- Engine 序列化与加载

### ⏱ 时间安排
- 3h：环境就绪（CUDA+TensorRT）  
- 3h：ONNX→TRT Engine 转换  
- 2h：Python 推理脚本  
- 2h：LeetCode（双指针 + 滑窗）

### 🖥️ 命令示例
```bash
trtexec --onnx=mlp.onnx --saveEngine=mlp_fp16.engine --fp16
# INT8（需要校准数据集与校准 cache）
trtexec --onnx=mlp.onnx --saveEngine=mlp_int8.engine --int8 --calib=/path/to/cache
```

### 📂 交付成果
- `mlp_fp16.engine` / `mlp_int8.engine`  
- `tensorrt_infer.py`  
- `report_day4.md`（数据表：精度/时延/吞吐）

### 🔍 深度追问
1. FP16 提速的硬件根因（Tensor Core）？对带宽与算力的影响？  
2. INT8 校准数据分布如何影响精度与延迟？  
3. tactic 选择受哪些约束（算子/shape/内存/工作空间）？  
4. 动态 shape profile 不合理会导致什么后果（fallback/性能抖动）？  
5. Engine 反序列化/缓存如何减少冷启动成本？  
6. DLA 与 GPU 混合执行的调度策略/限制？  

### 🧪 实验
1. FP32/FP16/INT8 三套引擎：对比 P50/P95、吞吐、精度差异。  
2. workspace 限制从 64MB→1GB，观察 tactic 变化与性能曲线。  
3. INT8 标定集大小 50→500→5000，测精度/性能，比较 per-channel vs per-tensor。  
4. 动态 profile `(1,3,224,224)/(8,3,224,224)/(64,3,224,224)` 下，逐 batch 测吞吐。  
5. 记录 engine 反序列化时间与首批请求延迟。  

---

## **Day 5 - Triton Inference Server**

### 🎯 学习目标
- 在 Triton 部署 ONNX/TRT 模型  
- 使用 HTTP/gRPC Client 调用  
- 了解 Triton 指标端点

### 📌 学习重点
- 模型库结构 `models/<name>/<version>/model.onnx`  
- `config.pbtxt` 基本字段  
- HTTP/gRPC Client SDK 调用

### ⏱ 时间安排
- 3h：运行 tritonserver  
- 3h：准备模型库与配置文件  
- 2h：编写 client 代码  
- 2h：LeetCode（DFS + BFS）

### 🖥️ 命令示例
```bash
docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002   -v $PWD/models:/models nvcr.io/nvidia/tritonserver:23.05-py3   tritonserver --model-repository=/models
```

### 📂 交付成果
- `models/`（ONNX/TRT 双版本）  
- `config.pbtxt`  
- `client_infer.py`  
- `report_day5.md`

### 🔍 深度追问
1. `instance_group` 如何影响并发度与显存占用？  
2. 动态批处理的 `max_queue_delay_microseconds` 如何影响 P99？  
3. 序列化模型需要 `sequence_batching` 时有哪些注意点？  
4. Response Cache 与 Redis 缓存的 trade-off？  
5. `rate_limiter` 如何限制全局 QPS？  
6. `model_control_mode`（poll vs explicit）影响上线与回滚？  

### 🧪 实验
1. 同一模型：`instance_group count=1/2/4` 对吞吐/显存的影响。  
2. 打开/关闭 `dynamic_batching` 并设不同 `preferred_batch_size`，比较 P50/P99。  
3. HTTP vs gRPC 客户端：并发 64/256/1024 下延迟差异与 CPU 占用。  
4. 启用 response cache 或客户端 Redis 缓存，记录命中率与延迟改善。  
5. 切换 `model_control_mode=explicit`，模拟灰度上线，验证回滚。  

---

## **Day 6 - Prometheus + Grafana**

### 🎯 学习目标
- 安装 Prometheus/Grafana  
- 抓取 Triton 指标  
- 搭建 GPU/QPS/延迟 Dashboard

### 📌 学习重点
- scrape job 配置  
- Grafana 面板与告警规则  
- PromQL 查询

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
- `prometheus.yaml`  
- `grafana_dashboard.json`  
- `report_day6.md`

### 🔍 深度追问
1. `scrape_interval` 与目标端暴露延迟的匹配关系？  
2. 直方图桶设计不当会引发什么问题？  
3. Recording Rules 如何降低查询时 CPU 压力？  
4. 高基数标签对 TSDB 的影响与治理方案？  
5. `remote_write` 场景？  
6. Grafana 告警与 Alertmanager 的分工？  

### 🧪 实验
1. `scrape_interval` 设 1s/5s/15s，比较 CPU/内存与图表平滑度。  
2. 为延迟设计粗 vs 细 Histogram 桶，对比 P95/P99 稳定性与资源占用。  
3. 写一条 Recording Rule 聚合 QPS，压测下比较查询耗时。  
4. 构造 1w+ 标签 Gauge，观察 Prom 内存增长曲线并回滚。  
5. 配置 Grafana 告警，当 P99>阈值触发 webhook，验证通知链路。  

---

## **Day 7 - 项目整合**

### 🎯 学习目标
- 汇总前 6 天，形成可运行的推理平台雏形  
- 输出架构图 + README

### 📌 学习重点
- 目录结构/脚本化/可复现  
- 组件连通性：Docker→K8s→Triton→Prom/Grafana  
- 文档与演示

### ⏱ 时间安排
- 3h：代码整理  
- 3h：绘制架构图  
- 2h：编写 README  
- 2h：LeetCode（综合）

### 📂 交付成果
- `ai-infra-inference-service/`（含 Dockerfile/k8s/models/prometheus/grafana）  
- 架构图  
- `README.md`、`report_day7.md`

### 🔍 深度追问
1. 当前端到端瓶颈更可能在：序列化/网络/排队/GPU 计算/磁盘？证据？  
2. 系统在冷启动与热运行的性能差异？如何隐藏冷启动影响？  
3. 单模型 vs 多模型共存对吞吐/显存/调度的影响？  
4. HPA 基于 CPU vs 自定义指标（QPS/队列长度）的效果差异？  
5. Pod 失效/重启对 P99 的影响如何量化并写进 SLO？  
6. 回滚与限流预案是否完善？  

### 🧪 实验
1. 压测 100→2000 QPS，导出 P50/P95/P99 与队列曲线，定位拐点。  
2. 预热（warmup N 次）对首批请求延迟的改善量化。  
3. 同 GPU 部署两个模型（各2实例） vs 一个模型（4实例），对比吞吐/显存。  
4. 配置 HPA：①CPU 50%；②自定义 QPS 阈值，比较扩容响应时间与过度扩容。  
5. Chaos 实验：随机 delete pod，观测恢复时间与 P99 抖动。  


---


# 📘 Week 2 学习计划（合并加强版）

## **Day 8 - CUDA 基础**

### 🎯 学习目标
- 理解 CUDA 并行模型（thread/block/grid）  
- 编写向量加法 CUDA 程序  
- 配置 `nvcc` 编译环境

### 📌 学习重点
- kernel 启动参数与索引计算  
- blockDim/gridDim 的取值与吞吐  
- CPU↔GPU 数据流

### ⏱ 时间安排
- 3h：语法与执行模型  
- 2h：环境配置与 Hello CUDA  
- 3h：`vector_add.cu` 实作与对比  
- 2h：LeetCode（数组/前缀和）

### 🖥️ 代码示例
```cpp
__global__ void vector_add(const float* a,const float* b,float* c,int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) c[i]=a[i]+b[i];
}
```
```bash
nvcc -O2 vector_add.cu -o vec && ./vec
```

### 📂 交付成果
- `cuda_hello.cu`、`vector_add.cu`  
- `report_day8.md`（不同 blockDim 的性能表）

### 🔍 深度追问
1. grid-stride loop 相比一次性大 grid 的优劣？  
2. 线程索引计算溢出/越界的常见坑与防御式编程？  
3. register 压力如何影响 occupancy 与溢出到本地内存？  
4. block 维度（1D/2D/3D）选择的依据？  
5. `__launch_bounds__` 对编译器寄存器分配与性能的影响？  
6. `nvcc -O2/-O3` 与 `--use_fast_math` 的风险？  

### 🧪 实验
1. 用 grid-stride 写 `vector_add`，对比单发大 grid 的可扩展性。  
2. 人为制造越界访问，开启 `cuda-memcheck` 捕获错误。  
3. 逐步增大 blockDim，观察 register 使用与 occupancy 的关系（Nsight）。  
4. 1D/2D block 访问二维数组，比较可读性与 coalescing。  
5. `--use_fast_math` 开关对数值误差与吞吐的影响。  

---

## **Day 9 - CUDA 内存管理**

### 🎯 学习目标
- 掌握 global/shared/constant 内存特性  
- 完成矩阵加法 GPU 版  
- 观察不同内存模型带来的差异

### 📌 学习重点
- shared memory 生命周期与 bank  
- constant memory 广播优势  
- 访存模式与 coalescing

### ⏱ 时间安排
- 3h：各类内存语义与 API  
- 3h：`matrix_add.cu` 实作（global → shared）  
- 2h：bank 冲突实验  
- 2h：LeetCode（哈希/映射）

### 🖥️ 代码示例
```cpp
__global__ void mat_add(const float* A,const float* B,float* C,int N){
  int r = blockIdx.y*blockDim.y + threadIdx.y;
  int c = blockIdx.x*blockDim.x + threadIdx.x;
  if(r<N && c<N) C[r*N+c] = A[r*N+c] + B[r*N+c];
}
```

### 📂 交付成果
- `matrix_add_global.cu`、`matrix_add_shared.cu`  
- `report_day9.md`（带宽/时延对比与 bank 冲突截图）

### 🔍 深度追问
1. shared memory bank 冲突具体是怎么发生的？避免策略有哪些？  
2. constant memory 读取的广播机制与失效场景？  
3. texture memory 在采样/插值中的优势，何时优于 global？  
4. Unified Memory 如何迁移页面？过量使用会如何 thrash？  
5. `cudaMemcpyAsync` 与 stream 关联的前提？  
6. L2 缓存命中与 stride 访问关系？  

### 🧪 实验
1. `matrix_add`：global vs shared，两版性能与带宽估算。  
2. 设计 stride=17 访问触发 bank 冲突，再加 padding 消除，量化收益。  
3. 常量参数放 constant memory，测试命中优势。  
4. Unified Memory 超过显存容量，观察迁移/吞吐下降。  
5. `cudaMemcpyAsync` + 两个 stream，拷贝/计算重叠与否对比。  

---

## **Day 10 - CUDA 优化技巧**

### 🎯 学习目标
- 理解 warp divergence 与 memory coalescing  
- 使用 shared memory 优化矩阵乘  
- 用 Nsight 定位瓶颈并看 occupancy

### 📌 学习重点
- 分支发散对 warp 的影响  
- 合并访存（coalescing）  
- occupancy 与线程块大小关系

### ⏱ 时间安排
- 3h：优化理论与案例  
- 3h：发散/合并访实验  
- 2h：Nsight profiling  
- 2h：LeetCode（双指针）

### 📂 交付成果
- `mmul_naive.cu`、`mmul_shared.cu`、`mmul_coalesce.cu`  
- `nsight_report_day10.md`（occupancy/throughput 截图）

### 🔍 深度追问
1. 分支消除与查表法在 GPU 上的适用性？  
2. loop unrolling 的收益与 register 膨胀的权衡？  
3. 访存对齐与 coalescing 的关系？  
4. inline PTX 的价值与可维护性风险？  
5. occupancy 并非越高越好，何时会反向影响性能？  
6. Tensor Core 前置条件（shape/对齐/数据类型）？  

### 🧪 实验
1. 将 if-else 替换为 `selp` 或三目，对比性能。  
2. `#pragma unroll` 不同展开度与 register 使用/性能平衡点。  
3. AoS vs SoA 对 coalescing 的影响。  
4. `--maxrregcount` 限制寄存器，观察性能与溢出代价。  
5. 用 WMMA 接口实现小 GEMM，对比普通 shared tiling。  

---

## **Day 11 - CUDA 实战项目**

### 🎯 学习目标
- 完成 CPU vs GPU 的矩阵乘法对比  
- shared memory tile 优化  
- 调参找峰值点

### 📌 学习重点
- tile 大小选择  
- 寄存器/共享内存消耗与并发度  
- 计算/访存重叠

### ⏱ 时间安排
- 3h：CPU baseline 与正确性校验  
- 3h：GPU 优化（tiling + shared）  
- 2h：参数寻优与曲线图  
- 2h：LeetCode（数学/前缀）

### 📂 交付成果
- `mmul_cpu.cpp`、`mmul_gpu_tiled.cu`  
- `report_day11.md`（尺寸/配置→性能曲线）

### 🔍 深度追问
1. GEMM 的算强/算密度为何适合 GPU？  
2. tiling 尺寸如何与 SM 结构匹配？  
3. 双缓冲如何隐藏访存延迟？  
4. 与 cuBLAS 的差距来自哪里？  
5. 不同 N 下复杂度线性度验证？  
6. 数值稳定性与性能的平衡？  

### 🧪 实验
1. CPU vs GPU vs cuBLAS（sgemm）随 N 的曲线。  
2. TILE=16/32/64 的资源占用与性能对比。  
3. 双缓冲优化与无双缓冲对比。  
4. FP16 累加到 FP32 的误差/性能对比。  
5. Nsight 定位瓶颈并提出优化计划。  

---

## **Day 12 - TensorRT Plugin 入门**

### 🎯 学习目标
- 了解为什么/何时需要自定义 Plugin  
- 编写一个简单激活函数 Plugin  
- 在构建/执行时正确注册与使用

### 📌 学习重点
- IPluginV2 接口  
- serialize/deserialize 生命周期  
- workspace 与输入输出维度处理

### ⏱ 时间安排
- 3h：接口与样例阅读  
- 3h：实现 ReLU/Swish 类 Plugin  
- 2h：集成至构图并测试  
- 2h：LeetCode（栈/模拟）

### 📂 交付成果
- `my_relu_plugin.{h,cc,cu}`  
- `build_and_test.sh`  
- `report_day12.md`（功能/性能对比）

### 🔍 深度追问
1. `IPluginV2DynamicExt` vs `IPluginV2` 的选型标准？  
2. 支持的数据格式与 layout 转换的代价？  
3. plugin 的序列化兼容性如何保障？  
4. 多线程/多实例安全问题？  
5. shape 推断与动态维度边界检查？  
6. plugin 错误处理策略？  

### 🧪 实验
1. 实现 Swish/Mish Plugin，与原生算子延迟对比。  
2. 给 plugin 增加内部 cache，测首帧与后续帧差异。  
3. 动态 shape 输入下验证 dims 处理正确性。  
4. 序列化/反序列化稳定性测试。  
5. 并发 1/4/8 实例下的线程安全与吞吐对比。  

---

## **Day 13 - Prometheus 深入**

### 🎯 学习目标
- 编写自定义 Exporter  
- 设计合理的指标  
- 接入 Grafana 实时展示

### 📌 学习重点
- Counter/Gauge/Histogram/Summary 的适用边界  
- 高基数标签问题  
- Alertmanager 基础

### ⏱ 时间安排
- 3h：Exporter 开发  
- 3h：Prometheus 抓取与联调  
- 2h：Grafana 面板设计  
- 2h：LeetCode（哈希/堆）

### 📂 交付成果
- `exporter_gpu_temp.py`  
- `prometheus.yaml`  
- `report_day13.md`（抓取延迟/负载观察）

### 🔍 深度追问
1. Counter vs Gauge vs Histogram 的适用边界？  
2. Exemplars 如何帮助追踪高延迟请求？  
3. Service Discovery 与静态 target 的差异？  
4. `query_range` 大窗口查询优化？  
5. Alertmanager 路由树设计要点？  
6. 远端存储拓扑与成本考量？  

### 🧪 实验
1. 为延迟与 QPS 设计合理 Histogram 桶。  
2. 打通 exemplars 与 tracing 链路。  
3. 用 Service Discovery 自动抓取 k8s Pod，验证 relabeling。  
4. 写 Recording Rule 聚合 1 分钟 QPS，比较查询耗时。  
5. 配置 Alertmanager：不同路由发不同渠道，做一次演练。  

---

## **Day 14 - 周总结与项目增强**

### 🎯 学习目标
- 总结 Week1~2 所有实验与数据  
- 给平台加一条关键告警链路  
- 形成“能写进简历”的周度成果

### 📌 学习重点
- 数据沉淀：曲线/表格/图  
- 告警：GPU>90%、延迟>阈值  
- README：复现实验步骤

### ⏱ 时间安排
- 3h：整理报告与截图  
- 3h：告警规则与通知渠道  
- 2h：项目描述润色  
- 2h：LeetCode（综合）

### 📂 交付成果
- `alerts.yaml`  
- `weekly_report_wk2.md`  
- `resume_bullet_week2.txt`

### 🔍 深度追问
1. 本周关键 SLI 指标有哪些？  
2. SLO 目标与当前差距？优先改哪个？  
3. 限流/排队/降级如何设计？  
4. HPA/告警是否可能震荡？如何抑制？  
5. 变更回滚流程是否自动化？  
6. 下一周目标与验收标准？  

### 🧪 实验
1. 压测 + 限流，验证 P99 稳定性。  
2. 灰度发布 10%→50%→100%，记录延迟与错误率。  
3. 故障演练：kill Prom/Grafana/Pod，评估 MTTR。  
4. 成本评估：固定 QPS 下测不同实例/$/QPS。  
5. 输出周报：曲线+结论+下一步计划。  


---


# 📘 Week 3 学习计划（合并加强版）

## **Day 15 - TensorRT 高级优化**

### 🎯 学习目标
- 学习 TensorRT tactic 选择与优化流程  
- 掌握 workspace 与性能的关系  
- 分析 engine 序列化、缓存机制

### 📌 学习重点
- tactic 自动搜索与限制条件  
- workspace 大小对性能影响  
- engine cache 作用与部署优化

### ⏱ 时间安排
- 3h：阅读 TRT 优化文档  
- 3h：实验不同 workspace 配置  
- 2h：分析 engine cache 行为  
- 2h：LeetCode（贪心 + 动态规划）

### 📂 交付成果
- `tensorrt_tactic_test.py`  
- `report_day15.md`（性能与 workspace 对比）

### 🔍 深度追问
1. TensorRT tactic 选择过程依赖哪些维度？  
2. workspace 太小会导致什么？  
3. tactic cache 是否可跨设备/版本？  
4. 为什么有时需要禁用某些 tactic？  
5. TensorRT engine cache 的原理？  
6. 如何在生产环境减少 engine 构建时间？  

### 🧪 实验
1. 设置 workspace=64MB/256MB/1GB，比较性能。  
2. 开启/禁用部分 tactic，比较延迟。  
3. 使用 engine cache，比较冷启动延迟。  
4. 在不同 GPU 上加载同一 engine，验证兼容性。  
5. 压测动态 batch 下的性能。  

---

## **Day 16 - Redis 缓存接入 Triton**

### 🎯 学习目标
- 学习 Triton Response Cache 功能  
- 接入 Redis 作为外部缓存  
- 评估缓存对性能的提升

### 📌 学习重点
- Response Cache API  
- Redis 部署与连接  
- 缓存命中率与延迟改善

### ⏱ 时间安排
- 3h：部署 Redis  
- 3h：修改 client 接入 Redis 缓存  
- 2h：压测缓存前后性能  
- 2h：LeetCode（LRU 缓存相关）

### 📂 交付成果
- `redis_client.py`  
- `report_day16.md`（命中率与延迟对比）

### 🔍 深度追问
1. Response Cache 内部如何索引请求？  
2. 缓存命中率对 P99 延迟影响？  
3. 缓存过期策略如何选择？  
4. 缓存和 batch 是否冲突？  
5. Redis 高并发瓶颈在哪？  
6. 缓存和一致性问题如何处理？  

### 🧪 实验
1. 部署 Redis 并接入 Triton Response Cache。  
2. 记录 10%/50%/90% 命中率下的延迟。  
3. 设置过期时间 1s/10s/60s，比较结果。  
4. 压测并发 1k/5k 请求时 Redis CPU/延迟。  
5. 模拟不一致更新，验证数据冲突。  

---

## **Day 17 - Triton Ensemble 模型**

### 🎯 学习目标
- 使用 Ensemble 管理多模型推理流程  
- 配置推理 pipeline（前处理+模型+后处理）  
- 理解依赖与流水线执行机制

### 📌 学习重点
- Ensemble 配置文件  
- DAG 模型依赖关系  
- Pipeline 并行与调度

### ⏱ 时间安排
- 3h：写一个前处理+模型+后处理 pipeline  
- 3h：部署 Ensemble  
- 2h：压测性能  
- 2h：LeetCode（拓扑排序/图论）

### 📂 交付成果
- `ensemble_config.pbtxt`  
- `report_day17.md`

### 🔍 深度追问
1. Ensemble 的 DAG 如何解析？  
2. 前后处理在 CPU/GPU 的调度方式？  
3. Ensemble 和普通模型部署差别？  
4. 依赖过长 pipeline 对延迟的影响？  
5. 是否能并行执行部分子图？  
6. 如何监控 ensemble 内各子模型的耗时？  

### 🧪 实验
1. 写一个简单 pipeline：normalize→model→softmax。  
2. 压测 ensemble vs 单模型延迟。  
3. Ensemble 中 CPU 前处理耗时测试。  
4. 在 Ensemble 中插入自定义 Python 后处理，观察性能。  
5. 用 Grafana 分别监控各子模型时延。  

---

## **Day 18 - Triton 多模型并发**

### 🎯 学习目标
- 在 Triton 部署多模型并发服务  
- 测试多模型并行性能  
- 优化 GPU 资源分配

### 📌 学习重点
- 多模型调度机制  
- instance_group 映射关系  
- GPU 显存分配策略

### ⏱ 时间安排
- 3h：部署多模型  
- 3h：配置并发调度  
- 2h：压测性能  
- 2h：LeetCode（调度类）

### 📂 交付成果
- `multi_model_config.pbtxt`  
- `report_day18.md`

### 🔍 深度追问
1. Triton 如何决定哪个模型优先执行？  
2. 不同模型显存分配冲突怎么解决？  
3. instance_group count=2 对性能影响？  
4. 模型间上下文是否会相互影响？  
5. 并发模型如何保证公平调度？  
6. 如何监控单个模型的延迟分布？  

### 🧪 实验
1. 部署两个模型共享 GPU，观察延迟。  
2. 修改 instance_group=1/2/4，比较吞吐。  
3. 模拟大模型+小模型同时运行，观察抖动。  
4. 用 Prometheus 监控各模型延迟曲线。  
5. 在两个模型间施加限流，观察公平性。  

---

## **Day 19 - BERT 模型部署优化**

### 🎯 学习目标
- 部署 BERT 模型到 Triton  
- 优化 batch 与动态 shape  
- 使用 TensorRT 加速 Transformer

### 📌 学习重点
- BERT 模型转换流程  
- TensorRT 优化 Transformer 层  
- batch 配置策略

### ⏱ 时间安排
- 3h：准备并导出 BERT ONNX  
- 3h：TensorRT 转换与优化  
- 2h：部署到 Triton  
- 2h：LeetCode（字符串/动态规划）

### 📂 交付成果
- `bert.onnx`  
- `bert.engine`  
- `report_day19.md`

### 🔍 深度追问
1. BERT 模型结构对推理的挑战点？  
2. Transformer 层如何用 TensorRT 优化？  
3. batch=1 vs batch=16 的延迟差异？  
4. 动态 shape 对 cache 的影响？  
5. BERT 序列长度影响延迟和显存？  
6. 如何选择 FP16/INT8？  

### 🧪 实验
1. 导出并运行 BERT ONNX baseline。  
2. TensorRT 转换 BERT，比较延迟。  
3. batch=1/8/16 测延迟与吞吐。  
4. 序列长度 32/128/512 的性能对比。  
5. 对比 FP16 vs INT8 的效果。  

---

## **Day 20 - Kubernetes HPA 扩缩容**

### 🎯 学习目标
- 使用 HPA 自动扩缩 Triton Pod  
- 配置基于 CPU/自定义指标的扩缩容  
- 验证在负载变化下的弹性

### 📌 学习重点
- HPA 配置与 API  
- Prometheus Adapter  
- 扩缩容策略

### ⏱ 时间安排
- 3h：配置基于 CPU 的 HPA  
- 3h：接入 Prometheus 自定义指标  
- 2h：压测触发扩缩容  
- 2h：LeetCode（二分/贪心）

### 📂 交付成果
- `hpa.yaml`  
- `report_day20.md`

### 🔍 深度追问
1. HPA 通过什么控制循环扩缩容？  
2. CPU vs QPS 指标扩容差别？  
3. 冷启动对延迟的影响？  
4. 扩容过快/过慢的风险？  
5. HPA 与 VPA 的区别？  
6. 如何避免扩容抖动？  

### 🧪 实验
1. 配置 CPU=50% 触发扩容。  
2. 用 Prometheus Adapter 配置 QPS 指标扩容。  
3. 压测 100→1000 QPS，观察扩容曲线。  
4. 冷启动下 P99 延迟曲线记录。  
5. 配置不同 cooldown，观察抖动差异。  

---

## **Day 21 - 周总结与系统复盘**

### 🎯 学习目标
- 总结 Week 3 的 Triton 高级特性与优化  
- 输出项目成果与实验数据  
- 形成复盘文档

### 📌 学习重点
- tactic/engine/cache  
- Redis 缓存  
- Ensemble、多模型并发  
- BERT 优化  
- HPA 扩缩容

### ⏱ 时间安排
- 3h：整理数据与实验对比表  
- 3h：复盘文档与架构图  
- 2h：简历项目 bullet point  
- 2h：LeetCode（综合）

### 📂 交付成果
- `weekly_report_wk3.md`  
- 架构图  
- 简历要点

### 🔍 深度追问
1. 本周实验的核心瓶颈是什么？  
2. 生产级部署还缺少哪些环节？  
3. HPA 的调优经验有哪些？  
4. 缓存的收益和局限？  
5. Ensemble 和多模型并发的适用场景？  
6. 下一阶段的优化目标？  

### 🧪 实验
1. 汇总所有实验结果做表格。  
2. 输出系统性能曲线图。  
3. 故障演练：随机删一个 Pod，记录恢复时间。  
4. 模拟 1k QPS 长时间运行，观察稳定性。  
5. 提炼三条可写简历的亮点。  


---


# 📘 Week 4 学习计划（合并加强版）

## **Day 22 - TensorRT 动态 Shape 优化**

### 🎯 学习目标
- 学习动态 shape 的 profile 配置方法  
- 理解 min/opt/max shape 的作用  
- 测试动态 batch 的性能差异

### 📌 学习重点
- 动态 shape profile 配置  
- Engine 针对动态 shape 的优化策略  
- batch 变化对性能的影响

### ⏱ 时间安排
- 3h：配置动态 shape profile  
- 3h：运行不同 batch 测试  
- 2h：比较延迟与吞吐曲线  
- 2h：LeetCode（数组与滑动窗口）

### 📂 交付成果
- `dynamic_profile.engine`  
- `report_day22.md`（batch-size 对比表格）

### 🔍 深度追问
1. 动态 shape 的本质是什么？  
2. min/opt/max 配置不合理会发生什么？  
3. 动态 shape 对 tactic 选择的影响？  
4. batch 增大延迟和吞吐的变化趋势？  
5. profile 数量与显存消耗关系？  
6. 如何在生产环境中选择 profile？  

### 🧪 实验
1. 配置 batch=1/16/64 三种 profile，运行测试。  
2. 记录延迟曲线与吞吐曲线。  
3. 修改 min/opt/max 为不合理值，观察性能下降。  
4. 增加多个 profile，监控显存占用。  
5. 比较动态 shape 与固定 shape 的差异。  

---

## **Day 23 - CUDA Stream 并行**

### 🎯 学习目标
- 学习 CUDA Stream 基本用法  
- 掌握计算与拷贝重叠  
- 实现多流并发

### 📌 学习重点
- Stream 创建与同步  
- 异步拷贝与 kernel 并行  
- 多流并发的瓶颈

### ⏱ 时间安排
- 3h：编写 stream 示例程序  
- 3h：实现 memcpy+kernel overlap  
- 2h：分析多流性能  
- 2h：LeetCode（并发模拟题）

### 📂 交付成果
- `cuda_stream_test.cu`  
- `report_day23.md`（并行效果截图）

### 🔍 深度追问
1. CUDA Stream 的调度机制？  
2. 默认 stream 与非默认 stream 的区别？  
3. overlap 的前提条件是什么？  
4. stream 数量过多会导致什么？  
5. 如何利用 event 进行同步？  
6. stream 优化与 pipeline 的关系？  

### 🧪 实验
1. 实现 memcpy+kernel overlap，对比性能。  
2. 单流 vs 多流并发性能。  
3. 使用 event 控制同步，观察效果。  
4. 开启过多 stream，观察性能瓶颈。  
5. Nsight 分析多流执行时间轴。  

---

## **Day 24 - NCCL 通信优化**

### 🎯 学习目标
- 学习 NCCL 基本通信 API  
- 使用 AllReduce 进行多卡通信  
- 测试通信带宽与延迟

### 📌 学习重点
- NCCL init 与通信模型  
- AllReduce/AllGather/ReduceScatter  
- GPU 间通信拓扑

### ⏱ 时间安排
- 3h：编写 NCCL 通信示例  
- 3h：运行多 GPU AllReduce  
- 2h：分析带宽与延迟  
- 2h：LeetCode（并查集/图论）

### 📂 交付成果
- `nccl_allreduce.cu`  
- `report_day24.md`（带宽延迟数据）

### 🔍 深度追问
1. NCCL 如何发现 GPU 拓扑？  
2. NVLink 与 PCIe 带宽差异？  
3. AllReduce 算法 Ring vs Tree？  
4. 通信与计算重叠如何实现？  
5. NCCL 调度与 CUDA Stream 的关系？  
6. 网络带宽对分布式性能影响？  

### 🧪 实验
1. 编写 NCCL AllReduce demo。  
2. 比较 2/4/8 卡延迟。  
3. NVLink vs PCIe 下的带宽差异。  
4. 使用多个 stream 进行通信+计算重叠。  
5. 记录通信曲线并绘制图表。  

---

## **Day 25 - Triton 模型热更新**

### 🎯 学习目标
- 学习 Triton 的 model_control_mode  
- 实现模型热加载与卸载  
- 测试热更新对服务影响

### 📌 学习重点
- model_control_mode: none/poll/explicit  
- 模型热更新机制  
- 热更新对请求的影响

### ⏱ 时间安排
- 3h：实验不同 control_mode  
- 3h：执行模型热加载/卸载  
- 2h：压测热更新期间性能  
- 2h：LeetCode（模拟/调度）

### 📂 交付成果
- `model_repo/`（含多版本模型）  
- `report_day25.md`

### 🔍 深度追问
1. Triton 如何检测模型文件变化？  
2. 显式加载/卸载的优缺点？  
3. 热更新期间请求如何处理？  
4. 模型版本切换策略？  
5. 如何保证回滚快速可靠？  
6. 热更新监控指标有哪些？  

### 🧪 实验
1. 配置 control_mode=poll，自动检测更新。  
2. 设置 control_mode=explicit，手动加载卸载。  
3. 在热更新期间压测，记录延迟。  
4. 切换不同版本模型，测试回滚。  
5. 监控日志与 Prometheus 指标。  

---

## **Day 26 - Istio 基础与流量治理**

### 🎯 学习目标
- 学习 Istio Gateway/VirtualService 基本配置  
- 实现流量分流与路由控制  
- 理解灰度发布与流量镜像

### 📌 学习重点
- Gateway/VirtualService/DestinationRule  
- 灰度发布与金丝雀部署  
- 流量镜像与回放

### ⏱ 时间安排
- 3h：安装 Istio  
- 3h：配置 Gateway/VirtualService  
- 2h：流量分流实验  
- 2h：LeetCode（图/最短路）

### 📂 交付成果
- `istio-gateway.yaml`  
- `report_day26.md`

### 🔍 深度追问
1. Istio 如何拦截流量？  
2. VirtualService 的匹配规则？  
3. DestinationRule 如何实现负载均衡？  
4. 灰度发布的风险点？  
5. 流量镜像的常见用途？  
6. Istio 对延迟的开销？  

### 🧪 实验
1. 配置 90% v1/10% v2 的灰度发布。  
2. 压测 v1/v2，记录延迟。  
3. 配置流量镜像，观察副本请求。  
4. 使用不同负载均衡策略，比较性能。  
5. 测试 Istio sidecar 对延迟的影响。  

---

## **Day 27 - CI/CD 实践 (ArgoCD/Jenkins)**

### 🎯 学习目标
- 学习 GitOps 工作流  
- 使用 ArgoCD 部署 Triton  
- 配置自动化流水线

### 📌 学习重点
- ArgoCD Application 配置  
- Git 推送触发自动部署  
- 回滚与版本管理

### ⏱ 时间安排
- 3h：安装 ArgoCD  
- 3h：配置 Application 与 GitOps 流程  
- 2h：实验自动化部署  
- 2h：LeetCode（字符串/栈）

### 📂 交付成果
- `argo-application.yaml`  
- `report_day27.md`

### 🔍 深度追问
1. GitOps 与手动 kubectl apply 的区别？  
2. ArgoCD 如何检测 Git 仓库变化？  
3. 自动化回滚流程如何实现？  
4. 与 Jenkins 的协同模式？  
5. Git 分支策略如何影响 CI/CD？  
6. 部署失败的回滚策略？  

### 🧪 实验
1. 配置 ArgoCD Application，连接 Git 仓库。  
2. 推送一次代码，触发自动部署。  
3. 模拟错误配置，观察回滚。  
4. 集成 Jenkins 触发构建+ArgoCD 部署。  
5. 测试版本切换与回滚时间。  

---

## **Day 28 - 周总结与项目增强**

### 🎯 学习目标
- 总结 Week 4 Triton+Istio+CI/CD 成果  
- 形成完整的推理服务交付链路  
- 输出文档与简历亮点

### 📌 学习重点
- TensorRT 动态 shape  
- CUDA Stream/NCCL  
- Triton 热更新  
- Istio 灰度与镜像  
- ArgoCD GitOps

### ⏱ 时间安排
- 3h：整理报告  
- 3h：绘制架构图  
- 2h：撰写简历亮点  
- 2h：LeetCode（综合）

### 📂 交付成果
- `weekly_report_wk4.md`  
- 架构图  
- 简历要点

### 🔍 深度追问
1. 本周实验的性能瓶颈是什么？  
2. 灰度发布与回滚流程是否完善？  
3. CI/CD 流水线自动化程度？  
4. Istio 开销可接受吗？  
5. NCCL 多卡通信的扩展性？  
6. 下一阶段优化方向？  

### 🧪 实验
1. 汇总本周所有实验数据。  
2. 绘制整体架构图。  
3. Chaos 实验：随机断开 Redis/模型，验证恢复。  
4. 长时间压测，验证稳定性。  
5. 输出简历可用亮点。  


---


# 📘 Week 5 学习计划（合并加强版）

## **Day 29 - TensorRT 多引擎管理**

### 🎯 学习目标
- 学习在同一服务中加载多个 TensorRT Engine  
- 理解多引擎管理机制  
- 分析不同引擎的性能差异

### 📌 学习重点
- 多引擎加载与调度  
- Engine 映射与选择逻辑  
- 多模型场景下的显存管理

### ⏱ 时间安排
- 3h：实现多引擎加载  
- 3h：测试多引擎性能差异  
- 2h：显存占用分析  
- 2h：LeetCode（哈希表/映射题）

### 📂 交付成果
- `multi_engine_test.py`  
- `report_day29.md`

### 🔍 深度追问
1. TensorRT 如何区分不同引擎？  
2. 多引擎切换的开销？  
3. 不同 batch 的引擎如何选择？  
4. 多引擎部署下显存分配策略？  
5. 引擎缓存能否多版本共存？  
6. 如何保证引擎切换时不中断服务？  

### 🧪 实验
1. 部署两个不同精度的引擎（FP16/INT8），比较性能。  
2. 动态选择 batch=1 与 batch=16 的引擎，记录切换延迟。  
3. 同时加载多个引擎，观察显存占用。  
4. 配置引擎缓存，验证多版本共存。  
5. 压测切换频繁时的延迟波动。  

---

## **Day 30 - Triton Python Backend**

### 🎯 学习目标
- 使用 Triton Python Backend 实现自定义逻辑  
- 在推理前后增加预处理/后处理  
- 分析 Python Backend 的性能开销

### 📌 学习重点
- Python Backend 接口  
- 输入输出 Tensor 管理  
- 性能瓶颈与优化策略

### ⏱ 时间安排
- 3h：编写 Python Backend 模块  
- 3h：部署到 Triton  
- 2h：压测性能  
- 2h：LeetCode（字符串处理题）

### 📂 交付成果
- `model.py`（Python Backend）  
- `report_day30.md`

### 🔍 深度追问
1. Python Backend 的启动方式？  
2. Python Backend 与 C++ Backend 的性能差距？  
3. Python Backend 适合什么场景？  
4. 内存拷贝是否为性能瓶颈？  
5. Python GIL 是否影响多实例并发？  
6. 如何优化 Python Backend 性能？  

### 🧪 实验
1. 实现一个 tokenize 的 Python Backend。  
2. 对比 Python Backend 与 C++ 性能。  
3. 部署 batch=1 与 batch=16，比较延迟。  
4. 开启多个实例，观察 Python GIL 影响。  
5. 使用 PyPy 或 Cython 优化部分逻辑，比较结果。  

---

## **Day 31 - Triton Custom Backend (C++)**

### 🎯 学习目标
- 学习编写 Triton C++ Backend  
- 掌握自定义逻辑的高性能实现  
- 对比 Python Backend 与 C++ Backend

### 📌 学习重点
- C++ Backend API  
- 内存管理与 Tensor 处理  
- 部署与调试方法

### ⏱ 时间安排
- 3h：阅读 Backend API 文档  
- 3h：实现简单自定义算子  
- 2h：部署并压测性能  
- 2h：LeetCode（位运算题）

### 📂 交付成果
- `custom_backend.cc`  
- `report_day31.md`

### 🔍 深度追问
1. C++ Backend 的执行模型？  
2. 内存管理如何避免拷贝？  
3. 与 Python Backend 的差异？  
4. 自定义 Backend 的调试难点？  
5. 如何保证 ABI 兼容性？  
6. 什么时候必须用 C++ Backend？  

### 🧪 实验
1. 实现 ReLU 的 C++ Backend。  
2. 与 Python Backend ReLU 比较延迟。  
3. 部署 batch=1/16，比较性能。  
4. 使用不同编译优化等级，观察性能差异。  
5. 多实例并发压测，比较性能。  

---

## **Day 32 - Triton 性能 Profiling**

### 🎯 学习目标
- 学习 Triton 的性能分析工具  
- 使用 perf_analyzer 压测模型  
- 输出性能曲线

### 📌 学习重点
- perf_analyzer 工具  
- 吞吐/延迟/显存使用  
- 分析性能瓶颈

### ⏱ 时间安排
- 3h：使用 perf_analyzer 压测  
- 3h：收集性能指标  
- 2h：绘制性能曲线  
- 2h：LeetCode（二分/查找题）

### 📂 交付成果
- `perf_report_day32.md`  
- 性能曲线图

### 🔍 深度追问
1. perf_analyzer 的采样方式？  
2. 吞吐与延迟的平衡点？  
3. batch size 对性能曲线影响？  
4. P50/P95/P99 指标的差异？  
5. 如何识别系统瓶颈？  
6. 如何利用 perf_analyzer 优化配置？  

### 🧪 实验
1. 对单模型压测，收集延迟/吞吐。  
2. batch=1/16/64 的性能曲线。  
3. 启动多个实例，比较结果。  
4. P50/P95/P99 曲线绘制。  
5. 修改配置文件，观察性能变化。  

---

## **Day 33 - TensorRT 动态 Batch 优化**

### 🎯 学习目标
- 学习 TensorRT 动态 batch 配置  
- 对比固定 batch 与动态 batch 性能  
- 优化推理延迟与吞吐

### 📌 学习重点
- 动态 batch profile  
- Batch 配置策略  
- 性能优化

### ⏱ 时间安排
- 3h：配置动态 batch profile  
- 3h：压测不同 batch 配置  
- 2h：对比固定 batch 与动态 batch  
- 2h：LeetCode（数学题）

### 📂 交付成果
- `dynamic_batch.engine`  
- `report_day33.md`

### 🔍 深度追问
1. 动态 batch 的本质是什么？  
2. 配置不合理会造成什么？  
3. batch 增大对显存的影响？  
4. 动态 batch 对延迟/吞吐的改善？  
5. 如何选择合适 batch？  
6. 动态 batch 与 dynamic shape 的关系？  

### 🧪 实验
1. 配置 batch=1/16/64 profile，压测性能。  
2. 对比固定 batch 与动态 batch。  
3. 修改 batch 配置不合理值，观察性能。  
4. 分析显存占用随 batch 的变化。  
5. 绘制延迟与吞吐曲线。  

---

## **Day 34 - Triton 模型版本管理**

### 🎯 学习目标
- 学习 Triton 模型版本管理机制  
- 部署多版本模型并切换  
- 理解回滚流程

### 📌 学习重点
- 模型目录结构  
- 版本切换策略  
- 回滚与灰度发布

### ⏱ 时间安排
- 3h：部署多版本模型  
- 3h：切换不同版本  
- 2h：测试回滚性能  
- 2h：LeetCode（链表题）

### 📂 交付成果
- `model_repo/`（多版本）  
- `report_day34.md`

### 🔍 深度追问
1. Triton 如何识别多版本模型？  
2. 默认加载最新版本的机制？  
3. 灰度发布如何实现？  
4. 回滚流程如何保证稳定？  
5. 多版本共存的资源占用？  
6. 如何在大规模部署中管理模型版本？  

### 🧪 实验
1. 部署 v1/v2 两个版本模型。  
2. 压测切换版本时的延迟。  
3. 模拟回滚操作，观察性能。  
4. 多版本共存时显存占用。  
5. 灰度发布策略下的流量分配。  

---

## **Day 35 - 周总结与系统增强**

### 🎯 学习目标
- 总结 Week 5 的 Triton 与 TensorRT 优化  
- 输出实验结果与架构改进  
- 形成简历亮点

### 📌 学习重点
- 多引擎管理  
- Python/C++ Backend  
- perf_analyzer 性能分析  
- 动态 batch  
- 模型版本管理

### ⏱ 时间安排
- 3h：整理实验数据  
- 3h：撰写周报与架构图  
- 2h：简历亮点提炼  
- 2h：LeetCode（综合题）

### 📂 交付成果
- `weekly_report_wk5.md`  
- 架构图  
- 简历要点

### 🔍 深度追问
1. 多引擎/多版本管理的最佳实践？  
2. Python Backend 与 C++ Backend 的 trade-off？  
3. perf_analyzer 的关键指标？  
4. 动态 batch 的收益？  
5. 生产部署中的风险点？  
6. 下一阶段的优化目标？  

### 🧪 实验
1. 汇总所有实验数据做表格。  
2. 绘制性能对比曲线。  
3. 灰度切换实验，验证可靠性。  
4. 长时间运行压测，观察稳定性。  
5. 输出简历亮点三条。  


---


# 📘 Week 6 学习计划（合并加强版）

## **Day 36 - TensorRT INT8 进阶优化**

### 🎯 学习目标
- 深入理解 INT8 量化与校准流程  
- 掌握不同校准算法的差异  
- 实践 per-tensor 与 per-channel 量化

### 📌 学习重点
- INT8 校准原理  
- Entropy、MinMax、Percentile 校准方法  
- 精度与性能平衡

### ⏱ 时间安排
- 3h：阅读 INT8 文档与 API  
- 3h：实现不同校准方法  
- 2h：对比精度与延迟  
- 2h：LeetCode（数学与数值类）

### 📂 交付成果
- `int8_calibration.py`  
- `report_day36.md`

### 🔍 深度追问
1. INT8 校准的数学原理？  
2. Entropy 与 MinMax 的差别？  
3. per-tensor 与 per-channel 的 trade-off？  
4. 为什么需要代表性数据集？  
5. 不同输入分布对精度影响？  
6. 如何监控 INT8 精度回退？  

### 🧪 实验
1. 用 MinMax 校准生成引擎，记录精度。  
2. 改用 Entropy 校准，比较结果。  
3. per-tensor vs per-channel 对比精度/性能。  
4. 使用不同大小校准集（50/500/5000 样本），比较效果。  
5. 压测 INT8 与 FP16、FP32 的延迟差异。  

---

## **Day 37 - CUDA Graphs 优化**

### 🎯 学习目标
- 学习 CUDA Graphs 的基本原理  
- 使用 CUDA Graphs 进行推理加速  
- 对比常规 kernel launch 的性能

### 📌 学习重点
- CUDA Graphs API  
- kernel 批量 launch 优化  
- 适用场景与限制

### ⏱ 时间安排
- 3h：阅读 CUDA Graphs 示例  
- 3h：实现一个推理 CUDA Graph  
- 2h：比较性能差异  
- 2h：LeetCode（图论基础）

### 📂 交付成果
- `cuda_graph_infer.cu`  
- `report_day37.md`

### 🔍 深度追问
1. CUDA Graphs 的本质是什么？  
2. 为什么能减少 launch overhead？  
3. Graph Capture 的限制条件？  
4. 动态 shape 下能否使用 CUDA Graphs？  
5. CUDA Graphs 与 Stream 的关系？  
6. 如何在深度学习推理中应用？  

### 🧪 实验
1. 用常规 kernel launch 实现推理 baseline。  
2. 用 CUDA Graphs 优化相同推理，比较延迟。  
3. 修改 batch=1/16/64，比较差异。  
4. 捕获 Graph 后多次复用，测试加速比。  
5. 分析 Nsight 时间轴。  

---

## **Day 38 - Triton 模型 Ensemble Pipeline**

### 🎯 学习目标
- 学习 Triton Ensemble pipeline 配置  
- 在 pipeline 中实现前处理/后处理  
- 对比 pipeline 与单模型性能

### 📌 学习重点
- Ensemble DAG 配置  
- Pipeline 执行顺序与依赖  
- 性能监控

### ⏱ 时间安排
- 3h：编写 Ensemble 配置  
- 3h：部署前处理+模型+后处理 pipeline  
- 2h：压测 pipeline 性能  
- 2h：LeetCode（拓扑排序/依赖图）

### 📂 交付成果
- `ensemble_config.pbtxt`  
- `report_day38.md`

### 🔍 深度追问
1. Ensemble DAG 的解析方式？  
2. Pipeline 并行执行的条件？  
3. 前处理运行在 CPU/GPU 的区别？  
4. Ensemble 的瓶颈如何排查？  
5. 如何监控 pipeline 内各步骤时延？  
6. 适合 Ensemble 的场景？  

### 🧪 实验
1. 部署 normalize→模型→softmax pipeline。  
2. 压测 Ensemble vs 单模型延迟。  
3. 将前处理放 CPU vs GPU，对比延迟。  
4. 插入 Python 后处理，观察性能。  
5. Prometheus 监控各子模型耗时。  

---

## **Day 39 - Triton Sequence Batch**

### 🎯 学习目标
- 学习序列 batch 的应用场景（RNN/会话模型）  
- 理解 sequence ID 与控制信号  
- 测试 Sequence Batch 的性能

### 📌 学习重点
- sequence_batching 配置  
- start/end/ready flags  
- 有状态模型支持

### ⏱ 时间安排
- 3h：配置 sequence_batching  
- 3h：部署有状态模型  
- 2h：压测性能  
- 2h：LeetCode（队列题）

### 📂 交付成果
- `sequence_model_config.pbtxt`  
- `report_day39.md`

### 🔍 深度追问
1. Sequence Batch 如何保持上下文？  
2. start/end/ready 的作用？  
3. Sequence ID 的分配策略？  
4. 并发序列如何调度？  
5. sequence batch 的局限性？  
6. RNN 与 Transformer 对 sequence batch 的需求差异？  

### 🧪 实验
1. 部署一个简单的 RNN 模型，配置 sequence_batching。  
2. 模拟 10 条对话并发，观察效果。  
3. 压测 sequence batch vs 普通 batch 性能。  
4. 错误配置 ID，观察报错。  
5. 长对话场景下性能曲线。  

---

## **Day 40 - Triton 模型加权路由**

### 🎯 学习目标
- 学习 model ensemble 中的 conditional routing  
- 配置加权路由到多个模型  
- 测试多版本加权推理

### 📌 学习重点
- 加权路由配置  
- 多版本模型分流  
- 性能与公平性

### ⏱ 时间安排
- 3h：编写 weighted routing 配置  
- 3h：部署多版本模型加权推理  
- 2h：压测性能与公平性  
- 2h：LeetCode（概率/随机化）

### 📂 交付成果
- `weighted_routing_config.pbtxt`  
- `report_day40.md`

### 🔍 深度追问
1. Triton 如何实现加权路由？  
2. 加权路由与灰度发布的关系？  
3. 路由策略如何影响延迟？  
4. 多模型并发时的公平性？  
5. 加权路由的监控指标？  
6. 适合用加权路由的场景？  

### 🧪 实验
1. 配置 70% v1/30% v2 加权路由。  
2. 压测公平性与延迟。  
3. 切换比例 50/50，观察差异。  
4. 灰度发布场景测试回滚。  
5. Grafana 监控分流比例。  

---

## **Day 41 - Kubernetes DaemonSet/StatefulSet**

### 🎯 学习目标
- 学习 DaemonSet 与 StatefulSet 应用场景  
- 部署日志收集与缓存服务  
- 理解 StatefulSet 的稳定标识

### 📌 学习重点
- DaemonSet 原理与应用  
- StatefulSet PVC 管理  
- 区别与适用场景

### ⏱ 时间安排
- 3h：编写 DaemonSet 日志采集配置  
- 3h：部署 StatefulSet 应用  
- 2h：测试扩容与滚动更新  
- 2h：LeetCode（数组模拟）

### 📂 交付成果
- `daemonset.yaml`、`statefulset.yaml`  
- `report_day41.md`

### 🔍 深度追问
1. DaemonSet 为什么适合日志/监控 Agent？  
2. StatefulSet Pod 命名规则？  
3. StatefulSet 的持久化卷绑定逻辑？  
4. 扩容/缩容时的稳定性？  
5. 与 Deployment 的差异？  
6. 在推理服务中适用场景？  

### 🧪 实验
1. 部署 DaemonSet 收集节点日志。  
2. 部署 StatefulSet MySQL，观察 PVC。  
3. 扩容 StatefulSet，验证 Pod 命名稳定性。  
4. 滚动更新 StatefulSet，观察顺序。  
5. 比较 StatefulSet 与 Deployment 差异。  

---

## **Day 42 - 周总结与系统优化**

### 🎯 学习目标
- 总结 Week 6 Triton+CUDA+K8s 的优化  
- 整理实验数据与结果  
- 形成简历亮点

### 📌 学习重点
- INT8 优化  
- CUDA Graphs  
- Ensemble Pipeline  
- Sequence Batch  
- 加权路由  
- DaemonSet/StatefulSet

### ⏱ 时间安排
- 3h：整理实验数据  
- 3h：撰写周报与架构图  
- 2h：简历亮点提炼  
- 2h：LeetCode（综合题）

### 📂 交付成果
- `weekly_report_wk6.md`  
- 架构图  
- 简历要点

### 🔍 深度追问
1. INT8 优化的收益与限制？  
2. CUDA Graphs 的适用边界？  
3. Ensemble Pipeline 的瓶颈？  
4. Sequence Batch 适合什么场景？  
5. DaemonSet 与 StatefulSet 在生产中的应用？  
6. 下一阶段的优化目标？  

### 🧪 实验
1. 汇总本周实验数据并做表格。  
2. 绘制性能与延迟曲线。  
3. Chaos 实验：随机删除 StatefulSet Pod，验证恢复。  
4. 长时间压测，验证稳定性。  
5. 输出简历亮点三条。  


---


# 📘 Week 7 学习计划（合并加强版）

## **Day 43 - Kubernetes Service Mesh (Istio 高级)**

### 🎯 学习目标
- 理解 Istio 的高级功能  
- 学习流量治理、熔断、限流策略  
- 掌握请求追踪与指标收集

### 📌 学习重点
- VirtualService 高级配置  
- 熔断/限流策略实现  
- Istio 与 Prometheus/Grafana 集成

### ⏱ 时间安排
- 3h：配置熔断与限流  
- 3h：启用请求追踪  
- 2h：监控数据收集  
- 2h：LeetCode（图/路径题）

### 📂 交付成果
- `istio-advanced.yaml`  
- `report_day43.md`

### 🔍 深度追问
1. 熔断策略在 Istio 如何配置？  
2. 限流与 HPA 扩缩容的关系？  
3. 请求追踪如何与 Jaeger/Zipkin 集成？  
4. Istio 对延迟的额外开销？  
5. Prometheus 如何采集 Istio 指标？  
6. Istio 的 Sidecar Injection 机制？  

### 🧪 实验
1. 配置 1000 QPS 下的熔断策略，观察效果。  
2. 设置限流=500 QPS，压测比较。  
3. 开启 Jaeger，记录一次请求链路。  
4. Grafana 展示 Istio 请求延迟分布。  
5. 比较开启 Istio 与未开启的性能差异。  

---

## **Day 44 - Kubernetes Operator 模式**

### 🎯 学习目标
- 理解 Operator 模式  
- 学习自定义控制器原理  
- 编写一个简单的 Operator

### 📌 学习重点
- CRD 与 Operator 关系  
- Controller 的调谐循环  
- Operator 框架（Kubebuilder/Operator SDK）

### ⏱ 时间安排
- 3h：编写简单 CRD  
- 3h：实现自定义控制器  
- 2h：部署并测试 Operator  
- 2h：LeetCode（设计题）

### 📂 交付成果
- `crd.yaml`、`operator.go`  
- `report_day44.md`

### 🔍 深度追问
1. CRD 与内置资源的区别？  
2. Operator 如何实现自动化？  
3. Reconcile 循环的核心逻辑？  
4. 何时需要写自定义 Operator？  
5. Kubebuilder 与 Operator SDK 的差异？  
6. Operator 的监控与告警？  

### 🧪 实验
1. 编写一个管理 ConfigMap 的 Operator。  
2. 部署 CRD 并创建资源，观察 Operator 行为。  
3. 故意修改资源，验证自动调谐。  
4. 扩展 Operator 增加指标导出。  
5. 压测 Operator 并发处理能力。  

---

## **Day 45 - 分布式训练 vs 推理架构对比**

### 🎯 学习目标
- 理解分布式训练与推理架构差异  
- 学习训练数据并行/模型并行  
- 对比推理场景的优化重点

### 📌 学习重点
- 数据并行 vs 模型并行  
- 参数服务器 vs AllReduce  
- 推理的低延迟需求

### ⏱ 时间安排
- 3h：阅读分布式训练论文/框架  
- 3h：对比推理架构  
- 2h：绘制对比图  
- 2h：LeetCode（数学/组合）

### 📂 交付成果
- `training_vs_inference.md`  
- 架构对比图

### 🔍 深度追问
1. 数据并行与模型并行的优缺点？  
2. 参数服务器模式的瓶颈？  
3. AllReduce 如何避免通信热点？  
4. 推理为什么更关注延迟？  
5. 分布式推理的挑战？  
6. 未来趋势：统一训练+推理架构？  

### 🧪 实验
1. 实现数据并行的简单 demo。  
2. 模拟参数服务器瓶颈。  
3. AllReduce 与参数服务器对比性能。  
4. 推理延迟曲线绘制。  
5. 输出架构对比表。  

---

## **Day 46 - GPU 监控与告警**

### 🎯 学习目标
- 学习 GPU 监控的关键指标  
- 使用 DCGM exporter 接入 Prometheus  
- 设置告警规则

### 📌 学习重点
- GPU 利用率/显存/温度指标  
- Prometheus 抓取 DCGM metrics  
- Grafana Dashboard 与告警

### ⏱ 时间安排
- 3h：安装 DCGM exporter  
- 3h：配置 Prometheus 抓取  
- 2h：Grafana 面板与告警规则  
- 2h：LeetCode（实现监控数据结构题）

### 📂 交付成果
- `prometheus-dcgm.yaml`  
- `grafana_gpu_dashboard.json`  
- `report_day46.md`

### 🔍 深度追问
1. GPU 哪些指标最关键？  
2. 高温如何影响 GPU 性能？  
3. 显存碎片化的危害？  
4. DCGM exporter 工作原理？  
5. 如何配置告警避免噪音？  
6. GPU 指标与推理性能的关系？  

### 🧪 实验
1. 部署 DCGM exporter。  
2. Prometheus 抓取 GPU 指标。  
3. Grafana 面板展示 GPU 利用率。  
4. 配置高温告警阈值=85℃。  
5. 压测推理，观察 GPU 指标变化。  

---

## **Day 47 - GPU 多进程服务 (MPS)**

### 🎯 学习目标
- 学习 CUDA MPS 工作原理  
- 在单 GPU 上运行多个进程共享资源  
- 分析 MPS 的性能收益

### 📌 学习重点
- CUDA MPS Daemon  
- GPU 上下文共享  
- 多进程并发调度

### ⏱ 时间安排
- 3h：配置并启动 MPS  
- 3h：运行多进程推理任务  
- 2h：分析性能数据  
- 2h：LeetCode（并发题）

### 📂 交付成果
- `mps_test.sh`  
- `report_day47.md`

### 🔍 深度追问
1. MPS 如何共享 GPU 上下文？  
2. 为什么 MPS 可以减少 context 切换开销？  
3. MPS 的局限？  
4. 与 MIG 的区别？  
5. 如何监控 MPS 的运行情况？  
6. 哪些应用最适合用 MPS？  

### 🧪 实验
1. 启动 MPS Daemon 并运行两个进程。  
2. 记录开启/关闭 MPS 的性能差异。  
3. 模拟高并发小请求场景。  
4. Prometheus 监控 MPS GPU 使用率。  
5. 对比 MPS 与非 MPS 下 P99 延迟。  

---

## **Day 48 - GPU 多实例 (MIG)**

### 🎯 学习目标
- 学习 NVIDIA MIG 技术原理  
- 将一张 GPU 划分为多个实例  
- 部署多实例推理服务

### 📌 学习重点
- MIG 分区方式  
- MIG 与 MPS 的区别  
- MIG 的调度策略

### ⏱ 时间安排
- 3h：配置 MIG 分区  
- 3h：部署多实例推理服务  
- 2h：压测多实例性能  
- 2h：LeetCode（模拟/划分题）

### 📂 交付成果
- `mig_config.sh`  
- `report_day48.md`

### 🔍 深度追问
1. MIG 如何实现物理隔离？  
2. MIG 与 MPS 的差异？  
3. MIG 对延迟的改善？  
4. MIG 的限制？  
5. 多实例间是否共享显存？  
6. MIG 在云平台的应用？  

### 🧪 实验
1. 配置 1x7g vs 7x1g MIG 模式。  
2. 部署 7 个小模型，观察性能。  
3. 压测单实例与多实例性能对比。  
4. 监控显存利用率与延迟曲线。  
5. 对比 MIG 与 MPS 的收益。  

---

## **Day 49 - 周总结与系统复盘**

### 🎯 学习目标
- 总结 Week 7 GPU 资源管理与服务网格成果  
- 输出实验数据与复盘文档  
- 提炼简历亮点

### 📌 学习重点
- Istio 流量治理  
- Operator 模式  
- 分布式训练 vs 推理对比  
- GPU 监控告警  
- MPS/MIG

### ⏱ 时间安排
- 3h：整理数据与对比表  
- 3h：撰写复盘文档  
- 2h：绘制架构图  
- 2h：LeetCode（综合题）

### 📂 交付成果
- `weekly_report_wk7.md`  
- 架构图  
- 简历要点

### 🔍 深度追问
1. Istio 的治理能力对推理架构的价值？  
2. Operator 在生产中的适用范围？  
3. GPU 监控告警的价值？  
4. MPS 与 MIG 的选择标准？  
5. GPU 资源调度的优化点？  
6. 下一阶段学习方向？  

### 🧪 实验
1. 汇总实验数据并绘制对比表。  
2. 输出系统复盘文档。  
3. 绘制 GPU 资源调度架构图。  
4. 长时间压测，观察稳定性。  
5. 提炼三条简历亮点。  


---


# 📘 Week 8 学习计划（合并加强版）

## **Day 50 - TensorRT 多卡推理**

### 🎯 学习目标
- 学习 TensorRT 在多 GPU 环境下的部署方式  
- 掌握多卡推理的调度策略  
- 测试多卡推理的性能提升

### 📌 学习重点
- 多 GPU 引擎构建  
- 显存分配与上下文管理  
- 多卡负载均衡策略

### ⏱ 时间安排
- 3h：配置多 GPU 环境  
- 3h：运行多卡推理实验  
- 2h：分析性能曲线  
- 2h：LeetCode（并行/负载均衡）

### 📂 交付成果
- `multi_gpu_engine.py`  
- `report_day50.md`

### 🔍 深度追问
1. TensorRT 如何在多 GPU 环境中调度？  
2. 多 GPU 引擎是否需要分别构建？  
3. 多卡推理的瓶颈点在哪里？  
4. 显存分配策略如何设计？  
5. 多 GPU 负载均衡如何实现？  
6. 数据拷贝是否是性能瓶颈？  

### 🧪 实验
1. 构建两个 GPU 的推理服务。  
2. 比较单 GPU vs 多 GPU 性能。  
3. 部署 batch=1 与 batch=64，观察加速比。  
4. 压测大模型在多 GPU 下的表现。  
5. 监控 GPU 利用率曲线。  

---

## **Day 51 - Triton 多实例多 GPU 部署**

### 🎯 学习目标
- 学习 Triton 在多 GPU 下的部署方法  
- 配置 instance_group 跨 GPU  
- 分析多实例性能差异

### 📌 学习重点
- instance_group 跨 GPU 配置  
- GPU affinity 与调度策略  
- 多实例对显存与吞吐的影响

### ⏱ 时间安排
- 3h：配置多 GPU Triton 服务  
- 3h：运行多实例推理  
- 2h：分析吞吐与延迟  
- 2h：LeetCode（多线程/调度）

### 📂 交付成果
- `multi_gpu_triton.yaml`  
- `report_day51.md`

### 🔍 深度追问
1. instance_group 如何指定 GPU？  
2. 多实例与多 GPU 的关系？  
3. 显存分配是否会冲突？  
4. 多 GPU 部署的瓶颈在哪里？  
5. 如何实现跨 GPU 的动态调度？  
6. 多 GPU 与多实例如何权衡？  

### 🧪 实验
1. 配置两个 GPU 的 instance_group。  
2. 部署两个模型共享 GPU，比较性能。  
3. 配置单模型多实例，观察延迟。  
4. 修改 instance_group=1/2/4，比较效果。  
5. 压测多 GPU 并发性能曲线。  

---

## **Day 52 - Kubernetes GPU Scheduling**

### 🎯 学习目标
- 学习 K8s GPU 调度机制  
- 使用 device-plugin 暴露 GPU 资源  
- 实现 Pod 与 GPU 绑定

### 📌 学习重点
- GPU 资源发现机制  
- Pod requests/limits 配置  
- GPU 亲和性调度

### ⏱ 时间安排
- 3h：阅读 device-plugin 文档  
- 3h：部署 GPU Pod  
- 2h：配置亲和性调度  
- 2h：LeetCode（图/匹配问题）

### 📂 交付成果
- `gpu_pod.yaml`  
- `report_day52.md`

### 🔍 深度追问
1. device-plugin 如何向 kubelet 注册资源？  
2. requests/limits 如何影响调度？  
3. Pod 如何绑定特定 GPU？  
4. GPU 节点打 taint 的作用？  
5. Topology Manager 的意义？  
6. Pod 与 GPU 映射如何监控？  

### 🧪 实验
1. 部署一个 GPU Pod，验证可用性。  
2. 修改 requests=1/2/4，观察调度变化。  
3. 配置 nodeSelector 将 Pod 绑定到特定节点。  
4. 为 GPU 节点打 taint，测试 tolerations。  
5. 查看 Pod describe 输出的 GPU 信息。  

---

## **Day 53 - GPU 拓扑与 NUMA 优化**

### 🎯 学习目标
- 学习 GPU 拓扑结构  
- 理解 NUMA 与 GPU 亲和性  
- 优化 GPU 与 CPU 绑定策略

### 📌 学习重点
- nvidia-smi topo 输出解析  
- NUMA 节点与 GPU 关系  
- CPU/GPU 亲和性优化

### ⏱ 时间安排
- 3h：分析 GPU 拓扑结构  
- 3h：配置 CPU 亲和性  
- 2h：测试性能差异  
- 2h：LeetCode（树/图题）

### 📂 交付成果
- `topo_report.md`  
- `report_day53.md`

### 🔍 深度追问
1. nvidia-smi topo 输出如何解读？  
2. PCIe 与 NVLink 对通信的影响？  
3. NUMA 与 GPU 的亲和性如何配置？  
4. CPU 绑定是否能提升性能？  
5. GPU 跨 NUMA 访问的代价？  
6. 如何在 K8s 中利用拓扑信息？  

### 🧪 实验
1. 使用 `nvidia-smi topo -m` 分析拓扑。  
2. 将进程绑定到不同 CPU 核，观察性能差异。  
3. 测试跨 NUMA 与本地 NUMA 的延迟。  
4. 在 K8s 中使用 Topology Manager 配置策略。  
5. 输出拓扑优化报告。  

---

## **Day 54 - Triton 动态批处理优化**

### 🎯 学习目标
- 学习 Triton 动态批处理机制  
- 配置 preferred_batch_size 与 max_queue_delay  
- 优化吞吐与延迟平衡

### 📌 学习重点
- dynamic_batching 配置  
- 延迟/吞吐权衡  
- 不同场景下的调优策略

### ⏱ 时间安排
- 3h：修改 dynamic_batching 配置  
- 3h：运行不同 batch 测试  
- 2h：分析延迟/吞吐曲线  
- 2h：LeetCode（队列题）

### 📂 交付成果
- `dynamic_batch_config.pbtxt`  
- `report_day54.md`

### 🔍 深度追问
1. dynamic_batching 的本质是什么？  
2. preferred_batch_size 如何选择？  
3. max_queue_delay 对 P99 延迟影响？  
4. batch size 与 GPU 利用率的关系？  
5. dynamic_batching 的局限？  
6. 如何在生产环境调优？  

### 🧪 实验
1. 配置 preferred_batch_size=4/8/16，测试延迟与吞吐。  
2. 修改 max_queue_delay=1ms/10ms/100ms，观察 P99。  
3. 绘制 batch size 与 GPU 利用率曲线。  
4. 对比 dynamic_batching 与固定 batch。  
5. 长时间压测，观察稳定性。  

---

## **Day 55 - Prometheus 联邦与长时存储**

### 🎯 学习目标
- 学习 Prometheus 联邦集群模式  
- 接入 Thanos/VM 作为长时存储  
- 测试大规模指标存储性能

### 📌 学习重点
- 联邦抓取模式  
- Thanos 架构与组件  
- 长时存储优化

### ⏱ 时间安排
- 3h：部署 Prometheus 联邦集群  
- 3h：接入 Thanos 存储  
- 2h：测试查询性能  
- 2h：LeetCode（哈希/并查集）

### 📂 交付成果
- `prometheus-federation.yaml`  
- `thanos-config.yaml`  
- `report_day55.md`

### 🔍 深度追问
1. 联邦抓取的工作机制？  
2. Thanos 如何做数据压缩？  
3. 联邦与 remote_write 的差异？  
4. Thanos Querier 的作用？  
5. 大规模存储的瓶颈在哪里？  
6. 如何降低存储成本？  

### 🧪 实验
1. 部署两级 Prometheus 联邦。  
2. 压测单点与联邦模式的性能。  
3. 接入 Thanos，比较查询耗时。  
4. 写入百万指标，观察磁盘与内存。  
5. 输出 Thanos 架构图与总结。  

---

## **Day 56 - 周总结与系统优化**

### 🎯 学习目标
- 总结 Week 8 GPU/K8s/Triton/监控优化成果  
- 输出实验数据与架构图  
- 形成简历亮点

### 📌 学习重点
- 多 GPU TensorRT  
- Triton 多实例  
- GPU 调度与拓扑优化  
- 动态批处理  
- Prometheus 联邦

### ⏱ 时间安排
- 3h：整理实验数据  
- 3h：撰写周报与架构图  
- 2h：简历亮点提炼  
- 2h：LeetCode（综合题）

### 📂 交付成果
- `weekly_report_wk8.md`  
- 架构图  
- 简历要点

### 🔍 深度追问
1. 多 GPU 部署的瓶颈？  
2. Triton 多实例的收益与代价？  
3. GPU 调度与 NUMA 优化的效果？  
4. 动态批处理的调优经验？  
5. Prometheus 联邦与 Thanos 的价值？  
6. 下一阶段目标？  

### 🧪 实验
1. 汇总实验数据并制表。  
2. 绘制 GPU 与 Triton 性能曲线。  
3. 压测多 GPU 长时间运行稳定性。  
4. 测试 Thanos 查询大窗口性能。  
5. 提炼简历亮点三条。  


---


# 📘 Week 9 学习计划（合并加强版）

## **Day 57 - 模型并行与流水线并行**

### 🎯 学习目标
- 学习模型并行与流水线并行原理  
- 掌握切分 Transformer 层的方法  
- 理解并行训练与推理的差异

### 📌 学习重点
- 张量并行与流水线并行  
- 通信/计算重叠  
- 推理时的并行调度

### ⏱ 时间安排
- 3h：阅读并行训练论文 Megatron-LM  
- 3h：实现一个简单的流水线并行 demo  
- 2h：分析推理并行的瓶颈  
- 2h：LeetCode（图/分治题）

### 📂 交付成果
- `pipeline_parallel_demo.py`  
- `report_day57.md`

### 🔍 深度追问
1. 模型并行的拆分粒度如何选择？  
2. 流水线并行的 bubble 开销是什么？  
3. Megatron-LM 如何做 tensor 并行？  
4. 推理 vs 训练在并行上的不同？  
5. 通信与计算的 overlap 如何实现？  
6. 并行策略如何影响显存利用率？  

### 🧪 实验
1. 将 4 层 Transformer 切分到两张 GPU。  
2. 测试流水线并行延迟。  
3. 模拟 tensor 并行的矩阵乘。  
4. 记录显存利用率变化。  
5. 对比数据并行与模型并行性能。  

---

## **Day 58 - 推理服务弹性伸缩**

### 🎯 学习目标
- 学习 Kubernetes HPA 与 KEDA 的弹性伸缩  
- 实现基于 QPS 的自动扩缩容  
- 测试在负载变化下的性能

### 📌 学习重点
- HPA 与 KEDA 的区别  
- Prometheus Adapter 与自定义指标  
- 冷启动问题

### ⏱ 时间安排
- 3h：配置 HPA 基于 QPS 扩缩容  
- 3h：实验 KEDA 事件驱动伸缩  
- 2h：压测性能  
- 2h：LeetCode（贪心/模拟题）

### 📂 交付成果
- `hpa_keda.yaml`  
- `report_day58.md`

### 🔍 深度追问
1. HPA 基于 CPU vs QPS 的差异？  
2. 冷启动对 P99 延迟影响？  
3. 扩容过快/过慢的风险？  
4. KEDA 如何支持事件驱动？  
5. 如何避免扩容抖动？  
6. 扩容对 GPU 资源的限制？  

### 🧪 实验
1. 配置 HPA 基于 QPS 扩缩容。  
2. 模拟流量从 100→1000 QPS，观察扩容。  
3. 测量冷启动前后的延迟。  
4. 实验不同 cooldown 配置的影响。  
5. 使用 KEDA Kafka 触发伸缩。  

---

## **Day 59 - GPU 高级调度策略**

### 🎯 学习目标
- 学习 Kubernetes 中的 GPU 调度优化  
- 使用 Topology Manager 进行优化  
- 测试多租户下的 GPU 调度效果

### 📌 学习重点
- GPU 亲和性与 NUMA 绑定  
- Topology Manager 策略  
- 公平调度与隔离

### ⏱ 时间安排
- 3h：研究 GPU 调度文档  
- 3h：配置 Topology Manager  
- 2h：测试公平性调度  
- 2h：LeetCode（调度题）

### 📂 交付成果
- `gpu_scheduler.yaml`  
- `report_day59.md`

### 🔍 深度追问
1. GPU 调度公平性如何实现？  
2. Topology Manager 如何优化性能？  
3. 多租户场景下的隔离？  
4. 显存碎片化如何避免？  
5. Pod 与 GPU 的强绑定风险？  
6. 如何在集群中保证 GPU 利用率？  

### 🧪 实验
1. 配置 GPU 亲和性 Pod。  
2. 实验 Topology Manager 的不同策略。  
3. 模拟两个用户共享 GPU 的场景。  
4. 观察公平性调度下的延迟。  
5. 显存碎片化场景测试。  

---

## **Day 60 - Triton 多模型调度优化**

### 🎯 学习目标
- 学习 Triton 的模型调度策略  
- 配置模型优先级与限流  
- 分析调度对性能的影响

### 📌 学习重点
- 模型优先级配置  
- 限流与 rate limiter  
- 调度公平性

### ⏱ 时间安排
- 3h：配置多模型调度  
- 3h：压测优先级与限流效果  
- 2h：分析公平性  
- 2h：LeetCode（优先队列）

### 📂 交付成果
- `multi_model_sched.yaml`  
- `report_day60.md`

### 🔍 深度追问
1. Triton 如何实现多模型调度？  
2. 模型优先级如何配置？  
3. 限流与 HPA 的关系？  
4. 调度公平性如何保证？  
5. 不同模型间是否会互相影响？  
6. 如何监控调度效果？  

### 🧪 实验
1. 部署两个模型，配置不同优先级。  
2. 压测优先级对延迟的影响。  
3. 配置 rate limiter 限制全局 QPS。  
4. 模拟高并发场景，观察公平性。  
5. Prometheus 监控调度效果。  

---

## **Day 61 - 模型压缩与蒸馏**

### 🎯 学习目标
- 学习模型压缩与蒸馏技术  
- 理解剪枝、量化、蒸馏的差异  
- 在推理中应用压缩技术

### 📌 学习重点
- 剪枝与稀疏化  
- 量化与蒸馏  
- 精度与性能权衡

### ⏱ 时间安排
- 3h：学习剪枝与量化方法  
- 3h：实验知识蒸馏  
- 2h：分析精度与延迟差异  
- 2h：LeetCode（动态规划/贪心）

### 📂 交付成果
- `distill_demo.py`  
- `report_day61.md`

### 🔍 深度追问
1. 剪枝的主要方法有哪些？  
2. 量化与蒸馏的适用场景？  
3. 蒸馏如何选择 teacher/student？  
4. 压缩对 Transformer 的挑战？  
5. 如何平衡精度与性能？  
6. 模型压缩的工程化难点？  

### 🧪 实验
1. 对一个小 CNN 进行剪枝。  
2. 使用 INT8 量化对比延迟。  
3. 实现一个蒸馏 demo。  
4. 比较不同蒸馏温度的效果。  
5. 绘制压缩前后精度与性能对比表。  

---

## **Day 62 - A/B 测试与灰度发布**

### 🎯 学习目标
- 学习 A/B 测试方法  
- 在 Triton/Istio 中实现灰度发布  
- 分析实验效果

### 📌 学习重点
- A/B 测试实验设计  
- 灰度发布配置  
- 实验指标收集

### ⏱ 时间安排
- 3h：设计 A/B 实验  
- 3h：配置 Istio 灰度路由  
- 2h：收集实验数据  
- 2h：LeetCode（概率/统计题）

### 📂 交付成果
- `ab_test_config.yaml`  
- `report_day62.md`

### 🔍 深度追问
1. A/B 测试需要哪些指标？  
2. 灰度发布如何避免用户感知？  
3. 样本量计算方法？  
4. Istio 如何实现流量分流？  
5. 实验失败如何回滚？  
6. 如何在生产环境安全执行 A/B 测试？  

### 🧪 实验
1. 配置 50% v1/50% v2 灰度发布。  
2. 收集性能与精度指标。  
3. 计算样本量对实验的影响。  
4. 模拟回滚操作。  
5. 输出实验对比表。  

---

## **Day 63 - 周总结与系统复盘**

### 🎯 学习目标
- 总结 Week 9 的并行/调度/优化实验  
- 输出完整的实验报告  
- 形成可放简历的项目亮点

### 📌 学习重点
- 模型并行与流水线并行  
- 弹性伸缩  
- GPU 高级调度  
- 多模型调度优化  
- 模型压缩与蒸馏  
- A/B 测试与灰度发布

### ⏱ 时间安排
- 3h：整理实验数据与图表  
- 3h：撰写复盘文档  
- 2h：绘制架构图  
- 2h：LeetCode（综合题）

### 📂 交付成果
- `weekly_report_wk9.md`  
- 架构图  
- 简历要点

### 🔍 深度追问
1. 本周最有价值的优化点是什么？  
2. GPU 调度的最佳实践？  
3. 模型压缩的适用场景？  
4. A/B 测试的风险控制？  
5. 弹性伸缩对生产系统的收益？  
6. 下一步的优化目标？  

### 🧪 实验
1. 汇总实验数据做表格。  
2. 输出系统复盘文档。  
3. 绘制 GPU 调度架构图。  
4. 长时间压测，验证稳定性。  
5. 提炼三条简历亮点。  


---


# 📘 Week 10 学习计划（合并加强版）

## **Day 64 - GPU 内核 Profiling**

### 🎯 学习目标
- 学习 GPU 内核 profiling 方法  
- 使用 Nsight Systems/Compute 分析瓶颈  
- 优化 kernel 性能

### 📌 学习重点
- Nsight Systems 时间轴分析  
- Nsight Compute kernel 级 profiling  
- Warp divergence / memory coalescing

### ⏱ 时间安排
- 3h：学习 Nsight Systems/Compute 用法  
- 3h：对已有 kernel 进行 profiling  
- 2h：优化并复测性能  
- 2h：LeetCode（模拟/实现题）

### 📂 交付成果
- `profiling_report_day64.md`  
- 优化后的 kernel 代码

### 🔍 深度追问
1. Nsight Systems 与 Nsight Compute 的区别？  
2. warp divergence 如何定位？  
3. memory coalescing 对性能的影响？  
4. occupancy 如何计算？  
5. 如何发现寄存器溢出？  
6. Nsight 中如何定位瓶颈？  

### 🧪 实验
1. 对 vector_add kernel 做 profiling。  
2. 观察 warp divergence 的情况。  
3. 修改访存方式，测试 coalescing 效果。  
4. 调整 blockDim，观察 occupancy。  
5. 对比优化前后的性能。  

---

## **Day 65 - GPU 高级算子优化**

### 🎯 学习目标
- 学习深度学习常用算子优化方法  
- 实现优化版矩阵乘/卷积  
- 理解 Tensor Core 的应用

### 📌 学习重点
- GEMM 优化方法  
- 卷积优化（im2col, Winograd）  
- Tensor Core API

### ⏱ 时间安排
- 3h：阅读 GEMM/卷积优化资料  
- 3h：实现优化算子  
- 2h：测试 Tensor Core 加速  
- 2h：LeetCode（矩阵题）

### 📂 交付成果
- `gemm_optimized.cu`、`conv_optimized.cu`  
- `report_day65.md`

### 🔍 深度追问
1. GEMM 优化的核心思想？  
2. 卷积如何降低计算复杂度？  
3. Winograd 算法的优势？  
4. Tensor Core 的要求与限制？  
5. CUDA kernel 如何调用 Tensor Core？  
6. cuBLAS/cuDNN 的优化点？  

### 🧪 实验
1. 实现 tiled GEMM。  
2. 测试 Winograd 卷积。  
3. 使用 Tensor Core 实现 GEMM。  
4. 对比 cuBLAS/cuDNN 性能。  
5. 记录优化前后性能。  

---

## **Day 66 - TensorRT Plugin 高级**

### 🎯 学习目标
- 深入学习 Plugin 生命周期  
- 支持动态 shape 与 batch  
- 优化 Plugin 性能

### 📌 学习重点
- IPluginV2DynamicExt 接口  
- serialize/deserialize 机制  
- Plugin 的多线程安全

### ⏱ 时间安排
- 3h：阅读高级 Plugin 文档  
- 3h：实现动态 shape Plugin  
- 2h：测试多实例并发性能  
- 2h：LeetCode（实现类题目）

### 📂 交付成果
- `custom_plugin.{h,cc,cu}`  
- `report_day66.md`

### 🔍 深度追问
1. IPluginV2DynamicExt 的意义？  
2. serialize/deserialize 的兼容性问题？  
3. Plugin 内存管理如何优化？  
4. Plugin 如何支持多线程？  
5. 动态 shape 的边界条件？  
6. Plugin 的调试方法？  

### 🧪 实验
1. 实现支持动态 shape 的 Plugin。  
2. 测试 batch=1/16/64 的性能。  
3. 多线程运行 Plugin，观察性能。  
4. 修改 serialize 格式，验证兼容性。  
5. 压测 Plugin 并发性能。  

---

## **Day 67 - Triton Python 后处理优化**

### 🎯 学习目标
- 优化 Triton Python Backend 的后处理逻辑  
- 减少 Python GIL 影响  
- 提升并发性能

### 📌 学习重点
- Python Backend GIL 问题  
- 多实例配置  
- Cython/Numba 优化方法

### ⏱ 时间安排
- 3h：阅读 Python Backend 优化资料  
- 3h：实现 Cython/Numba 优化后处理  
- 2h：压测性能  
- 2h：LeetCode（字符串处理）

### 📂 交付成果
- `optimized_postprocess.py`  
- `report_day67.md`

### 🔍 深度追问
1. Python GIL 对并发的影响？  
2. 多实例能否缓解 GIL？  
3. Cython 与 Numba 的差别？  
4. Python Backend 的优化边界？  
5. 如何与 C++ Backend 协同？  
6. 何时必须迁移到 C++？  

### 🧪 实验
1. 实现 Python 后处理 baseline。  
2. 使用 Cython 优化后处理。  
3. 使用 Numba 优化后处理。  
4. 配置多实例，比较效果。  
5. 对比 Python 与 C++ 后处理性能。  

---

## **Day 68 - Prometheus 高级优化**

### 🎯 学习目标
- 学习 Prometheus 的性能优化方法  
- 配置 Recording Rule/Alert 优化查询  
- 优化大规模场景下的存储

### 📌 学习重点
- Prometheus 存储优化  
- Recording Rules 应用  
- Thanos/VM 扩展

### ⏱ 时间安排
- 3h：配置 Recording Rules  
- 3h：实验大窗口查询优化  
- 2h：接入 Thanos  
- 2h：LeetCode（统计类）

### 📂 交付成果
- `prometheus_advanced.yaml`  
- `report_day68.md`

### 🔍 深度追问
1. Prometheus 查询的瓶颈？  
2. Recording Rules 的作用？  
3. Alert 配置的噪声控制？  
4. Thanos 的架构与扩展能力？  
5. 高基数标签的优化方法？  
6. Prometheus 在大规模集群的局限？  

### 🧪 实验
1. 编写 Recording Rules 优化查询。  
2. 压测大窗口查询性能。  
3. 配置告警规则并触发。  
4. 接入 Thanos，测试查询性能。  
5. 模拟高基数标签，观察性能下降。  

---

## **Day 69 - Kubernetes 节点自动伸缩 (Cluster Autoscaler)**

### 🎯 学习目标
- 学习 Cluster Autoscaler 原理  
- 配置节点级扩缩容  
- 测试负载变化下的弹性

### 📌 学习重点
- Cluster Autoscaler 与 HPA 的关系  
- 扩容与缩容逻辑  
- 冷启动影响

### ⏱ 时间安排
- 3h：部署 Cluster Autoscaler  
- 3h：模拟流量测试扩缩容  
- 2h：分析性能曲线  
- 2h：LeetCode（贪心/动态规划）

### 📂 交付成果
- `cluster_autoscaler.yaml`  
- `report_day69.md`

### 🔍 深度追问
1. Cluster Autoscaler 的原理？  
2. 与 HPA 的配合方式？  
3. 缩容如何避免中断？  
4. 冷启动对性能影响？  
5. 节点扩缩容的监控？  
6. 自动伸缩的风险？  

### 🧪 实验
1. 部署 Cluster Autoscaler。  
2. 模拟流量从 100→2000 QPS，观察扩缩容。  
3. 测试缩容时服务可用性。  
4. 记录扩缩容时间。  
5. 对比仅 HPA 的表现。  

---

## **Day 70 - 周总结与系统优化**

### 🎯 学习目标
- 总结 Week 10 的优化点  
- 输出复盘报告与简历亮点  
- 形成系统优化方案

### 📌 学习重点
- GPU profiling 与优化  
- 高级算子优化  
- TensorRT Plugin  
- Python Backend 优化  
- Prometheus 高级优化  
- Cluster Autoscaler

### ⏱ 时间安排
- 3h：整理实验与数据  
- 3h：撰写复盘文档  
- 2h：绘制架构图  
- 2h：LeetCode（综合题）

### 📂 交付成果
- `weekly_report_wk10.md`  
- 架构图  
- 简历要点

### 🔍 深度追问
1. GPU profiling 得到的瓶颈？  
2. 算子优化的难点？  
3. TensorRT Plugin 的价值？  
4. Python Backend 的边界？  
5. Prometheus 优化的经验？  
6. 自动伸缩的风险与收益？  

### 🧪 实验
1. 汇总 GPU profiling 数据。  
2. 绘制优化前后曲线。  
3. 输出复盘文档。  
4. 绘制架构图。  
5. 提炼简历亮点三条。  


---


# 📘 Week 11 学习计划（合并加强版）

## **Day 71 - 大规模推理架构综述**

### 🎯 学习目标
- 学习大规模推理架构的核心设计  
- 理解高可用、低延迟、可扩展的原则  
- 对比主流推理框架架构

### 📌 学习重点
- 高可用架构模式  
- 扩展性设计  
- 成本与性能权衡

### ⏱ 时间安排
- 3h：阅读推理架构论文/白皮书  
- 3h：调研主流框架架构  
- 2h：绘制对比图  
- 2h：LeetCode（系统设计类）

### 📂 交付成果
- `inference_architecture.md`  
- 架构对比图

### 🔍 深度追问
1. 大规模推理的瓶颈有哪些？  
2. 高可用如何实现？  
3. 成本与性能如何平衡？  
4. 各推理框架的设计差异？  
5. 推理架构的未来趋势？  
6. 云原生对推理架构的影响？  

### 🧪 实验
1. 总结三种主流架构的差异。  
2. 绘制对比图。  
3. 输出文档对比优缺点。  
4. 调研开源项目并记录。  
5. 提炼可写简历的要点。  

---

## **Day 72 - 分布式推理调度**

### 🎯 学习目标
- 学习分布式推理调度方法  
- 理解任务切分与调度策略  
- 实现一个简单的调度器

### 📌 学习重点
- 调度算法（Round Robin, Least Load）  
- 任务切分与结果合并  
- 监控与反馈

### ⏱ 时间安排
- 3h：阅读调度算法文档  
- 3h：实现简单调度器  
- 2h：压测对比调度策略  
- 2h：LeetCode（调度/优先队列）

### 📂 交付成果
- `distributed_scheduler.py`  
- `report_day72.md`

### 🔍 深度追问
1. 调度算法的优缺点？  
2. 任务切分的粒度如何选择？  
3. 如何避免单点瓶颈？  
4. 调度结果如何反馈优化？  
5. GPU 与 CPU 混合调度的挑战？  
6. 分布式推理与训练调度的差异？  

### 🧪 实验
1. 实现 Round Robin 调度器。  
2. 实现 Least Load 调度器。  
3. 模拟任务切分与合并。  
4. 压测两种调度方式。  
5. 输出对比报告。  

---

## **Day 73 - 混合精度推理 (FP16/BF16/INT8)**

### 🎯 学习目标
- 学习混合精度推理原理  
- 对比 FP32/FP16/BF16/INT8 性能与精度  
- 实现混合精度配置

### 📌 学习重点
- 混合精度配置方法  
- 精度与性能平衡  
- 不同硬件对混合精度的支持

### ⏱ 时间安排
- 3h：阅读混合精度推理文档  
- 3h：实验不同精度推理  
- 2h：对比结果  
- 2h：LeetCode（数值类）

### 📂 交付成果
- `mixed_precision_test.py`  
- `report_day73.md`

### 🔍 深度追问
1. FP16 的优势与风险？  
2. BF16 与 FP16 的差别？  
3. INT8 精度回退问题？  
4. 不同硬件对精度的支持？  
5. 混合精度的调度策略？  
6. 生产中如何选择精度？  

### 🧪 实验
1. 对比 FP32/FP16 性能与精度。  
2. 测试 BF16 与 FP16 差异。  
3. INT8 与 FP16 的混合精度配置。  
4. 绘制性能/精度对比曲线。  
5. 输出选择策略总结。  

---

## **Day 74 - 模型服务安全性**

### 🎯 学习目标
- 学习模型服务的安全风险  
- 配置身份认证与鉴权  
- 实现 TLS 加密传输

### 📌 学习重点
- 身份认证（JWT/OAuth2）  
- TLS 配置  
- 安全审计与日志

### ⏱ 时间安排
- 3h：配置认证服务  
- 3h：部署 TLS 证书  
- 2h：测试安全性  
- 2h：LeetCode（字符串/哈希）

### 📂 交付成果
- `auth_config.yaml`  
- `tls_cert_config.yaml`  
- `report_day74.md`

### 🔍 深度追问
1. 模型服务面临的主要风险？  
2. TLS 如何配置？  
3. JWT 与 OAuth2 的差异？  
4. 鉴权失败的处理方式？  
5. 安全审计的关键点？  
6. 性能与安全的平衡？  

### 🧪 实验
1. 配置 JWT 鉴权。  
2. 部署 TLS 证书。  
3. 模拟未认证请求，观察拦截。  
4. 配置审计日志。  
5. 测试认证对延迟的影响。  

---

## **Day 75 - 灰度发布与回滚策略**

### 🎯 学习目标
- 学习灰度发布方法  
- 配置回滚策略  
- 实现版本切换

### 📌 学习重点
- Istio 灰度发布  
- Triton 多版本切换  
- 回滚流程

### ⏱ 时间安排
- 3h：配置灰度发布  
- 3h：实验回滚策略  
- 2h：压测灰度发布性能  
- 2h：LeetCode（模拟题）

### 📂 交付成果
- `gray_release.yaml`  
- `report_day75.md`

### 🔍 深度追问
1. 灰度发布如何保证平滑？  
2. 回滚流程的关键点？  
3. 如何检测新版本异常？  
4. 灰度发布的风险？  
5. 如何监控发布效果？  
6. 版本切换的最佳实践？  

### 🧪 实验
1. 配置 90% v1/10% v2 流量。  
2. 压测灰度发布性能。  
3. 模拟 v2 异常，执行回滚。  
4. 观察 Prometheus 指标变化。  
5. 输出回滚总结。  

---

## **Day 76 - 生产环境可观测性**

### 🎯 学习目标
- 学习生产环境可观测性实践  
- 配置分布式 tracing  
- 集成日志/指标/追踪三合一

### 📌 学习重点
- OpenTelemetry  
- 日志/指标/追踪一体化  
- 可观测性最佳实践

### ⏱ 时间安排
- 3h：部署 OpenTelemetry  
- 3h：接入日志与 tracing  
- 2h：Grafana 展示全链路数据  
- 2h：LeetCode（日志处理题）

### 📂 交付成果
- `otel_config.yaml`  
- `report_day76.md`

### 🔍 深度追问
1. 可观测性的三大支柱？  
2. OpenTelemetry 的作用？  
3. 日志与 tracing 的区别？  
4. 如何减少可观测性开销？  
5. 告警与 tracing 的结合？  
6. 生产级可观测性的挑战？  

### 🧪 实验
1. 部署 OpenTelemetry Collector。  
2. 收集推理服务日志。  
3. 配置 tracing，查看调用链。  
4. Grafana 展示全链路数据。  
5. 输出可观测性总结。  

---

## **Day 77 - 周总结与系统复盘**

### 🎯 学习目标
- 总结 Week 11 的架构、安全与可观测性成果  
- 输出复盘报告  
- 提炼简历亮点

### 📌 学习重点
- 大规模推理架构  
- 分布式调度  
- 混合精度推理  
- 模型服务安全性  
- 灰度发布与回滚  
- 可观测性

### ⏱ 时间安排
- 3h：整理数据与实验结果  
- 3h：撰写复盘文档  
- 2h：绘制架构图  
- 2h：LeetCode（综合题）

### 📂 交付成果
- `weekly_report_wk11.md`  
- 架构图  
- 简历要点

### 🔍 深度追问
1. 大规模推理架构的核心优化点？  
2. 分布式调度的瓶颈？  
3. 混合精度的收益与风险？  
4. 模型服务的安全挑战？  
5. 灰度发布与回滚的经验？  
6. 可观测性实践的难点？  

### 🧪 实验
1. 汇总 Week 11 数据与实验。  
2. 绘制整体架构图。  
3. 输出复盘文档。  
4. 模拟一次异常回滚。  
5. 提炼三条简历亮点。  


---


# 📘 Week 12 学习计划（合并加强版）

## **Day 78 - 大模型推理框架综述**

### 🎯 学习目标
- 学习主流大模型推理框架 (vLLM, FasterTransformer, TensorRT-LLM)  
- 对比其架构与优化点  
- 理解 KV Cache 与高效推理方法

### 📌 学习重点
- KV Cache 工作原理  
- 序列并行与张量并行  
- 框架的性能优化点

### ⏱ 时间安排
- 3h：阅读框架文档  
- 3h：对比架构设计  
- 2h：绘制对比表  
- 2h：LeetCode（系统设计题）

### 📂 交付成果
- `llm_frameworks_comparison.md`  
- 架构对比图

### 🔍 深度追问
1. vLLM 的连续批处理 (continuous batching) 原理？  
2. FasterTransformer 的优化点？  
3. TensorRT-LLM 如何实现 KV Cache？  
4. 推理框架与硬件的适配关系？  
5. 序列并行与张量并行的区别？  
6. 哪个框架更适合在线推理？  

### 🧪 实验
1. 部署 vLLM 运行一个小模型。  
2. 部署 FasterTransformer 测试延迟。  
3. TensorRT-LLM 部署，观察性能。  
4. 对比不同框架的 QPS 与延迟。  
5. 输出对比表格。  

---

## **Day 79 - KV Cache 优化**

### 🎯 学习目标
- 深入理解 KV Cache 的作用  
- 学习 KV Cache 的存储优化策略  
- 测试不同 KV Cache 策略的性能

### 📌 学习重点
- KV Cache 存储方式  
- Sliding Window/分页策略  
- Cache 命中率与延迟

### ⏱ 时间安排
- 3h：学习 KV Cache 实现  
- 3h：实现 Sliding Window 策略  
- 2h：压测性能  
- 2h：LeetCode（LRU 缓存题）

### 📂 交付成果
- `kv_cache_test.py`  
- `report_day79.md`

### 🔍 深度追问
1. KV Cache 的存储代价？  
2. Sliding Window 如何实现？  
3. KV Cache 与显存的关系？  
4. Cache 命中率对延迟的影响？  
5. KV Cache 的回收策略？  
6. 多 GPU 下如何共享 KV Cache？  

### 🧪 实验
1. 实现 KV Cache baseline。  
2. 测试 Sliding Window 策略。  
3. 压测不同窗口大小对性能的影响。  
4. 显存占用曲线绘制。  
5. 对比命中率与延迟关系。  

---

## **Day 80 - 张量并行与序列并行**

### 🎯 学习目标
- 学习张量并行与序列并行原理  
- 理解通信开销与优化策略  
- 实现简单的并行 demo

### 📌 学习重点
- Tensor Parallel 与 Sequence Parallel  
- 通信/计算 overlap  
- 多卡同步机制

### ⏱ 时间安排
- 3h：学习并行论文  
- 3h：实现简单 demo  
- 2h：分析通信开销  
- 2h：LeetCode（图/并查集）

### 📂 交付成果
- `tensor_parallel_demo.py`  
- `report_day80.md`

### 🔍 深度追问
1. 张量并行如何切分矩阵乘？  
2. 序列并行的适用场景？  
3. 通信开销如何隐藏？  
4. NCCL 在并行中的作用？  
5. 张量并行对显存的影响？  
6. 如何结合流水线并行？  

### 🧪 实验
1. 实现 2-way 张量并行。  
2. 实现序列并行 demo。  
3. 对比单卡与多卡延迟。  
4. 绘制通信/计算时间分布。  
5. 输出优化总结。  

---

## **Day 81 - Prompt 优化与缓存**

### 🎯 学习目标
- 学习 Prompt 的优化方法  
- 实现 Prompt Cache 机制  
- 分析缓存对延迟的改善

### 📌 学习重点
- Prompt Cache 存储  
- Prompt 复用策略  
- Prompt 优化技巧

### ⏱ 时间安排
- 3h：阅读 Prompt Cache 机制  
- 3h：实现 Prompt Cache demo  
- 2h：测试缓存效果  
- 2h：LeetCode（字符串/哈希）

### 📂 交付成果
- `prompt_cache_demo.py`  
- `report_day81.md`

### 🔍 深度追问
1. Prompt Cache 的存储代价？  
2. Prompt 复用如何实现？  
3. Cache 命中率与延迟关系？  
4. Prompt 优化的技巧？  
5. 长 Prompt 如何优化？  
6. Prompt Cache 的一致性问题？  

### 🧪 实验
1. 实现 Prompt Cache baseline。  
2. 模拟高命中率场景。  
3. 测试低命中率性能。  
4. 记录缓存前后延迟。  
5. 输出总结报告。  

---

## **Day 82 - 长文本推理优化**

### 🎯 学习目标
- 学习长文本推理的挑战与优化方法  
- 测试分块推理与 Sliding Window  
- 分析延迟与显存关系

### 📌 学习重点
- 长文本分块策略  
- Sliding Window 推理  
- KV Cache 与长文本

### ⏱ 时间安排
- 3h：阅读长文本推理论文  
- 3h：实现分块推理 demo  
- 2h：测试长文本延迟  
- 2h：LeetCode（字符串/区间题）

### 📂 交付成果
- `long_context_demo.py`  
- `report_day82.md`

### 🔍 深度追问
1. 长文本推理的瓶颈？  
2. 分块策略如何选择？  
3. Sliding Window 的优缺点？  
4. KV Cache 对长文本的作用？  
5. 显存占用如何优化？  
6. 长文本推理的未来方向？  

### 🧪 实验
1. 实现分块推理 demo。  
2. 实现 Sliding Window 推理。  
3. 压测长文本 1k/4k/16k tokens。  
4. 绘制显存占用曲线。  
5. 输出优化总结。  

---

## **Day 83 - 高并发推理优化**

### 🎯 学习目标
- 学习高并发推理优化方法  
- 实现请求队列与动态批处理  
- 分析高并发下的稳定性

### 📌 学习重点
- 动态批处理与调度  
- 请求队列设计  
- 高并发稳定性优化

### ⏱ 时间安排
- 3h：实现请求队列 demo  
- 3h：测试动态批处理效果  
- 2h：高并发压测  
- 2h：LeetCode（队列/调度题）

### 📂 交付成果
- `high_concurrency_demo.py`  
- `report_day83.md`

### 🔍 深度追问
1. 高并发推理的瓶颈？  
2. 动态批处理如何实现？  
3. 请求队列如何设计？  
4. 高并发下如何保证稳定性？  
5. CPU 与 GPU 的分工？  
6. 如何在生产中实现弹性？  

### 🧪 实验
1. 实现请求队列 demo。  
2. 配置动态批处理。  
3. 压测 1k/5k/10k QPS。  
4. 记录延迟分布曲线。  
5. 输出总结报告。  

---

## **Day 84 - 周总结与系统优化**

### 🎯 学习目标
- 总结 Week 12 的大模型推理优化  
- 输出实验数据与复盘文档  
- 提炼简历亮点

### 📌 学习重点
- 推理框架对比  
- KV Cache 优化  
- 张量/序列并行  
- Prompt 优化  
- 长文本推理  
- 高并发优化

### ⏱ 时间安排
- 3h：整理实验数据  
- 3h：撰写复盘文档  
- 2h：绘制架构图  
- 2h：LeetCode（综合题）

### 📂 交付成果
- `weekly_report_wk12.md`  
- 架构图  
- 简历要点

### 🔍 深度追问
1. 本周最有效的优化点？  
2. 长文本推理的挑战？  
3. 高并发推理的瓶颈？  
4. 不同框架的差异？  
5. 如何在生产落地？  
6. 下一阶段目标？  

### 🧪 实验
1. 汇总本周实验数据做表格。  
2. 输出复盘文档。  
3. 绘制整体架构图。  
4. 长时间压测稳定性。  
5. 提炼三条简历亮点。  


---


# 📘 Week 13 学习计划（合并加强版）

## **Day 85 - 全局系统复盘 I**

### 🎯 学习目标
- 回顾 Week1~6 的实验与收获  
- 整理文档与架构图  
- 提炼系统优化经验

### 📌 学习重点
- Docker/K8s 基础  
- TensorRT/Triton 优化  
- Prometheus+Grafana 监控  
- CUDA 基础与优化

### ⏱ 时间安排
- 3h：整理 Week1~3 报告  
- 3h：整理 Week4~6 报告  
- 2h：绘制复盘架构图  
- 2h：LeetCode（综合题）

### 📂 交付成果
- `global_review_part1.md`  
- 架构图

### 🔍 深度追问
1. 前半程最大的学习瓶颈？  
2. 哪些实验最有价值？  
3. K8s 与容器化的收获？  
4. TensorRT 优化的要点？  
5. CUDA 的理解是否足够深入？  
6. 如何在面试中表述前半程收获？  

### 🧪 实验
1. 汇总 Week1~6 的数据表格。  
2. 绘制系统学习曲线。  
3. 输出架构复盘图。  
4. 提炼三条简历亮点。  
5. 输出经验总结文档。  

---

## **Day 86 - 全局系统复盘 II**

### 🎯 学习目标
- 回顾 Week7~12 的实验与收获  
- 整理文档与架构图  
- 提炼系统优化经验

### 📌 学习重点
- GPU 资源管理 (MPS/MIG)  
- 分布式推理与调度  
- 混合精度与模型压缩  
- 大模型推理优化 (KV Cache/并行)

### ⏱ 时间安排
- 3h：整理 Week7~9 报告  
- 3h：整理 Week10~12 报告  
- 2h：绘制复盘架构图  
- 2h：LeetCode（系统设计题）

### 📂 交付成果
- `global_review_part2.md`  
- 架构图

### 🔍 深度追问
1. 后半程最大的学习瓶颈？  
2. GPU 资源管理的收获？  
3. 分布式推理的理解是否到位？  
4. 模型压缩与蒸馏的收获？  
5. 大模型推理优化的难点？  
6. 如何在面试中表述后半程收获？  

### 🧪 实验
1. 汇总 Week7~12 的数据表格。  
2. 绘制 GPU 资源调度图。  
3. 输出系统复盘图。  
4. 提炼三条简历亮点。  
5. 输出经验总结文档。  

---

## **Day 87 - 项目最终整合 I**

### 🎯 学习目标
- 将前半程与后半程成果整合为统一项目  
- 搭建端到端推理平台  
- 输出最终 README 与架构图

### 📌 学习重点
- 项目目录与脚本化  
- Triton+TensorRT+K8s+监控整合  
- 复盘与演示

### ⏱ 时间安排
- 3h：整合代码与配置  
- 3h：搭建端到端 pipeline  
- 2h：绘制最终架构图  
- 2h：LeetCode（综合题）

### 📂 交付成果
- `ai_infra_final_project/`  
- 架构图  
- `README.md`

### 🔍 深度追问
1. 系统整合的难点？  
2. Triton 与 TensorRT 如何协同？  
3. K8s 的自动扩缩容是否稳定？  
4. 监控与告警是否完善？  
5. 性能瓶颈是否解决？  
6. 最终项目能否展示生产价值？  

### 🧪 实验
1. 整合所有模块，形成 pipeline。  
2. 压测端到端性能。  
3. 绘制最终架构图。  
4. 输出最终 README。  
5. 演示一次完整的推理请求。  

---

## **Day 88 - 项目最终整合 II**

### 🎯 学习目标
- 在项目中加入高阶功能 (灰度发布/缓存/并行优化)  
- 优化整体架构的健壮性  
- 形成生产级设计

### 📌 学习重点
- 灰度发布与回滚  
- KV Cache 与动态批处理  
- 高并发稳定性

### ⏱ 时间安排
- 3h：实现灰度发布与回滚  
- 3h：接入缓存与动态批处理  
- 2h：测试高并发场景  
- 2h：LeetCode（并发题）

### 📂 交付成果
- `final_project_config/`  
- `report_day88.md`

### 🔍 深度追问
1. 灰度发布的回滚流程是否完善？  
2. KV Cache 的命中率是否理想？  
3. 高并发下的稳定性如何？  
4. 监控能否捕捉关键异常？  
5. 架构是否具备容灾能力？  
6. 是否能写进生产经验？  

### 🧪 实验
1. 配置灰度发布。  
2. 配置 KV Cache。  
3. 压测 1k/5k/10k QPS。  
4. 模拟故障回滚。  
5. 输出总结报告。  

---

## **Day 89 - 简历亮点与面试准备 I**

### 🎯 学习目标
- 提炼三个月学习成果为简历亮点  
- 总结关键技术点  
- 准备面试问答

### 📌 学习重点
- Docker/K8s/Triton/TensorRT/CUDA  
- GPU 优化与并行  
- 监控与可观测性  
- 大模型推理优化

### ⏱ 时间安排
- 3h：撰写简历亮点  
- 3h：整理面试问题与答案  
- 2h：模拟问答  
- 2h：LeetCode（面试高频题）

### 📂 交付成果
- `resume_highlights.md`  
- `interview_questions.md`

### 🔍 深度追问
1. 面试中如何展示项目价值？  
2. 简历亮点如何提炼？  
3. 技术点如何串联为故事？  
4. 面试官可能的刁钻问题？  
5. 如何展现系统性思维？  
6. 面试中如何展示学习能力？  

### 🧪 实验
1. 输出三条简历亮点。  
2. 准备 20 道面试题。  
3. 模拟一次问答。  
4. 输出面试复盘文档。  
5. 提炼个人成长总结。  

---

## **Day 90 - 简历亮点与面试准备 II**

### 🎯 学习目标
- 深入优化简历与面试准备  
- 完成最终学习复盘  
- 输出学习闭环

### 📌 学习重点
- 简历亮点优化  
- 面试技巧总结  
- 学习闭环复盘

### ⏱ 时间安排
- 3h：优化简历内容  
- 3h：完善面试准备  
- 2h：输出学习复盘  
- 2h：LeetCode（综合题）

### 📂 交付成果
- `resume_final.md`  
- `final_review.md`

### 🔍 深度追问
1. 简历是否足够简洁有力？  
2. 面试故事是否完整？  
3. 学习闭环是否形成？  
4. 三个月的成长是否可量化？  
5. 如何展现未来规划？  
6. 是否达到入职 AI Infra 工程师水平？  

### 🧪 实验
1. 输出最终简历。  
2. 模拟三轮面试。  
3. 输出学习复盘文档。  
4. 总结三个月成长曲线。  
5. 提炼未来学习规划。  


---

