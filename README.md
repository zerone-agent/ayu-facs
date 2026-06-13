# AYU — FACS 人脸表情分析云服务

基于 OpenFace 3.0 的 FACS（面部动作编码系统）云服务，用于心理评估场景的面部 AU 检测与情绪映射。

## 架构

```
前端 MediaPipe 人脸裁剪 → WebSocket 224×224 帧 → 后端 AU 检测 → EMFACS 情绪映射 → 报告输出
```

- **FACS 引擎**：OpenFace 3.0（openface-test），纯 Python + PyTorch，CPU 单帧 ~38ms
- **情绪输出**：8 个 AU 强度 + 6 种基本情绪（OpenFace 原生 + EMFACS 规则融合）+ 效价/唤醒度
- **部署**：Docker Compose 一键部署，CPU 模式，支持 GPU 切换

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/analyze` | POST | 单图分析，上传图片返回 AU + 情绪 |
| `/` | GET | 测试页面（摄像头实时分析） |
| `/ws/stream` | WebSocket | 实时流式分析（224×224 RGBA 帧） |

### POST /analyze

```bash
# 本地
curl -X POST http://localhost:8001/analyze -F 'file=@assets/sample_face.jpg'

# 远程 ECS 示例
curl -X POST http://47.120.38.148:8001/analyze \
  -F 'file=@assets/sample_face.jpg' \
  -w "\nHTTP_CODE: %{http_code}\nTIME_TOTAL: %{time_total}s\n"
```

返回示例（基于 `assets/sample_face.jpg` 实测）：

```json
{
  "success": true,
  "frame_id": "e51a8860-b0f6-4a7a-8d40-99e2e456e66d",
  "timestamp": "2026-06-13T03:26:07.776523+00:00",
  "aus": {
    "AU01": 0.0005,
    "AU02": 0.0,
    "AU04": 0.0,
    "AU05": 0.0,
    "AU06": 0.0,
    "AU07": 0.0,
    "AU09": 0.1934,
    "AU10": 0.0
  },
  "emotions": {
    "Happiness": 0.0565,
    "Sadness": 0.0138,
    "Anger": 0.0081,
    "Fear": 0.0416,
    "Surprise": 0.0621,
    "Disgust": 0.062
  },
  "valence": -0.0037,
  "arousal": 0.1112
}
```

### WebSocket /ws/stream

发送 224×224×4 RGBA 原始字节，接收 JSON 分析结果。前端用 MediaPipe 裁剪人脸后推送。

## 部署

```bash
# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f
```

默认端口：宿主机 8001 → 容器 8000。

### GPU 模式

编辑 `docker-compose.yml`，取消注释 `deploy.resources` 段，并设置环境变量 `DEVICE=cuda`。

## 配置

通过环境变量配置（`docker-compose.yml`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEVICE` | `auto` | 推理设备：`cpu` / `cuda` / `auto` |
| `MAX_FPS` | `10` | WebSocket 流式分析最大帧率 |
| `OPENFACE_WEIGHTS_DIR` | `~/.openface/weights` | 模型权重目录 |

## 项目结构

```
├── src/
│   ├── config.py      # 设备配置、AU 列表、融合权重
│   ├── facs_engine.py # OpenFace 3.0 封装 + EMFACS 规则映射
│   └── server.py      # FastAPI 服务（/analyze, /ws/stream, /health）
├── assets/
│   └── sample_face.jpg # 示例测试图片
├── test_client.html   # 前端测试页面（MediaPipe + WebSocket）
├── Dockerfile         # Docker 构建（python:3.11-slim + CPU torch）
├── docker-compose.yml # 一键部署
├── requirements.txt   # Python 依赖
└── tests/             # EMFACS 规则测试（10/10 通过）
```

## 技术细节

- **OpenFace 3.0**：29.4M 参数，4.2 GFLOPs，单模型同时输出 AU（8类）+ 情绪（8类）+ 眼神 + 关键点
  - 8 个 AU：`AU01`（内眉提升）、`AU02`（外眉提升）、`AU04`（皱眉）、`AU06`（脸颊提升）、`AU09`（鼻皱）、`AU12`（嘴角上扬）、`AU25`（嘴唇分开）、`AU26`（下颌下垂）
- **EMFACS 规则**：将可用 AU 组合映射为 6 种基本情绪（Happiness/Sadness/Anger/Fear/Surprise/Disgust），已根据模型实际输出裁剪为 8 个 AU 子集
- **情绪融合**：OpenFace 原生情绪（权重 0.6）+ EMFACS 规则情绪（权重 0.4），可配置
- **模型权重**：来自 `nutPace/openface_weights`（HuggingFace），构建时通过国内镜像下载

## License

MIT
