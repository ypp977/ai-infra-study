# Day 6 — Prometheus + Grafana 保姆级教程

## 🎯 今日目标

1. 在 Docker 自定义网络下运行 **Triton + Prometheus + Grafana**
2. 配置 Prometheus 自动采集 Triton 指标
3. 在 Grafana 中（命令行配置）添加 Prometheus 数据源
4. 生成并导出一个监控面板 JSON

------

## 一、创建 Docker 网络

所有容器共享一个网络，互相可直接用容器名访问：

```bash
docker network create ai-monitor-net
```

## 二、启动 Triton

启动 **Day5 Triton** 并且 Triton 在本地 `8002` 端口暴露 **Prometheus Metrics**。

```bash
cd ../day5 && docker run -d --gpus all \
  --name triton \
  --network ai-monitor-net \
  -v $PWD/models:/models \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  nvcr.io/nvidia/tritonserver:23.05-py3 \
  tritonserver --model-repository=/models
```

验证 Triton metrics 是否可访问：

```bash
# 用 curl 检查 看到响应200 OK 并且加载了mnist_mlp模型 即可成功
curl -v http://localhost:8002/metrics | head -n 20
```

![image-20250827183501685](./report_day6.assets/image-20250827183501685.png)

## 三、配置 Prometheus

切换到`day6`目录创建配置目录：

```bash
mkdir -p monitor/prometheus
```

写配置文件 `monitor/prometheus/prometheus.yml`：

```yml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "triton"
    static_configs:
      - targets: ["triton:8002"]
```

这里直接用容器名 `triton`，因为它们在同一个网络。

------

## 四、启动 Prometheus

切换到`day6`目录启动容器

```bash
docker run -d \
  --name prometheus \
  --network ai-monitor-net \
  -v $PWD/monitor/prometheus:/etc/prometheus \
  -p 9090:9090 \
  prom/prometheus \
  --config.file=/etc/prometheus/prometheus.yml
```

验证 Prometheus 已经抓到 Triton 数据：

```bash
curl "http://localhost:9090/api/v1/targets" | jq .
```

看到 `"health": "up"` 表示成功。

![image-20250827210830446](./report_day6.assets/image-20250827210830446.png)

------

## 五、启动 Grafana

```bash
docker run -d \
  --name grafana \
  --network ai-monitor-net \
  -p 3000:3000 \
  grafana/grafana
```

默认用户名密码：`admin / admin`

因为你是命令行环境，可以用 `curl` 或 Grafana HTTP API 来配置。

------

## 六、命令行配置 Grafana

#### 1️⃣  登录 Grafana API（获取 token）

 直接用 admin/admin（如果没改过）：

```bash
curl -X POST http://admin:admin@localhost:3000/api/serviceaccounts \
  -H "Content-Type: application/json" \
  -d '{"name":"cli-service","role":"Admin"}'
```

![image-20250827211734798](./report_day6.assets/image-20250827211734798.png)

这里关键是 `"id":2`，我们接下来要用它来创建 token

#### 2️⃣ 给 Service Account 生成 Token

```bash
curl -X POST http://admin:admin@localhost:3000/api/serviceaccounts/2/tokens \
  -H "Content-Type: application/json" \
  -d '{"name":"cli-token"}'
```

响应中 `"key"` 就是你后续调用 API 用的 `<TOKEN>`。⚠️ 只会返回一次，要保存好。

![image-20250827212021405](./report_day6.assets/image-20250827212021405.png)

#### 3️⃣  添加 Prometheus 数据源

（Prometheus 容器和 Grafana 在同一个 `ai-monitor-net` 网络里）

```bash
curl -X POST http://localhost:3000/api/datasources \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer glsa_yQwCVCeqqJdOMPQIQEf2KFqX1MBPOoUd_390c4bdf" \
  -d '{
    "name":"Prometheus",
    "type":"prometheus",
    "url":"http://prometheus:9090",
    "access":"proxy",
    "basicAuth":false
  }'
```

![image-20250827212651377](./report_day6.assets/image-20250827212651377.png)

#### 4️⃣ 导入 Dashboard（监控 Triton）

保存为 `dashboard.json`：

```bash
{
  "dashboard": {
    "title": "Triton Monitoring CLI",
    "panels": [
      {
        "type": "timeseries",
        "title": "GPU Utilization",
        "targets": [{"expr": "nv_gpu_utilization"}]
      },
      {
        "type": "timeseries",
        "title": "QPS",
        "targets": [{"expr": "rate(nv_inference_request_success[1m])"}]
      },
      {
        "type": "timeseries",
        "title": "Inference Latency (ms)",
        "targets": [{"expr": "avg(nv_inference_request_duration_us)/1000"}]
      }
    ]
  },
  "overwrite": true
}
```

执行导入：

```bash
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer glsa_yQwCVCeqqJdOMPQIQEf2KFqX1MBPOoUd_390c4bdf" \
  -d @dashboard.json
```

![image-20250827213344087](./report_day6.assets/image-20250827213344087.png)

------

## 七、验证监控数据

1. 列出所有 Dashboards：

```bash
curl -H "Authorization: Bearer glsa_yQwCVCeqqJdOMPQIQEf2KFqX1MBPOoUd_390c4bdf" \
     http://localhost:3000/api/search
```

![image-20250827213455762](./report_day6.assets/image-20250827213455762.png)

2. 直接查询 Prometheus 指标（通过 Grafana 代理）：

```bash
curl -H "Authorization: Bearer glsa_yQwCVCeqqJdOMPQIQEf2KFqX1MBPOoUd_390c4bdf" \
     "http://localhost:3000/api/datasources/proxy/1/api/v1/query?query=nv_gpu_utilization"
```

![image-20250827213643083](./report_day6.assets/image-20250827213643083.png)

3. 登录web界面：

调用前面 Day5 的 `client_infer.py`

```bash
for i in {1..100}; do
  python3 client_infer.py > /dev/null
done
```

可以看到Grafana 面板里的**GPU利用率** **QPS 和延迟** 曲线就会动

![image-20250827215001074](./report_day6.assets/image-20250827215001074.png)
