# AYU — FACS 人脸表情分析云服务

基于 OpenFace 3.0 的 FACS（面部动作编码系统）云服务，用于心理评估场景的面部 AU 检测与情绪映射。

## 架构

```
前端 MediaPipe 人脸裁剪 → WebSocket 224×224 帧 → 后端 AU 检测 → EMFACS 情绪映射 → 报告输出
```

- **FACS 引擎**：OpenFace 3.0（openface-test），纯 Python + PyTorch，CPU 单帧 ~38ms
- **情绪输出**：20 个 AU 强度 + 6 种基本情绪（OpenFace 原生 + EMFACS 规则融合）+ 效价/唤醒度
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
curl -X POST http://localhost:8001/analyze -F 'file=@face.jpg'
```

返回示例：

```json
{
  "success": true,
  "aus": {"AU01": 0.0, "AU12": 0.85, ...},
  "emotions": {"Happiness": 0.72, "Sadness": 0.01, ...},
  "valence": 0.35,
  "arousal": 0.12
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
├── test_client.html   # 前端测试页面（MediaPipe + WebSocket）
├── Dockerfile         # Docker 构建（python:3.11-slim + CPU torch）
├── docker-compose.yml # 一键部署
├── requirements.txt   # Python 依赖
└── tests/             # EMFACS 规则测试（10/10 通过）
```

## 技术细节

- **OpenFace 3.0**：29.4M 参数，4.2 GFLOPs，单模型同时输出 AU（20类）+ 情绪（8类）+ 眼神 + 关键点
- **EMFACS 规则**：将 AU 组合映射为 6 种基本情绪（Happiness/Sadness/Anger/Fear/Surprise/Disgust）
- **情绪融合**：OpenFace 原生情绪（权重 0.6）+ EMFACS 规则情绪（权重 0.4），可配置
- **模型权重**：来自 `nutPace/openface_weights`（HuggingFace），构建时通过国内镜像下载

## License

MIT
