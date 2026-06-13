# AYU FACS 基准测试图片

本目录包含针对 `/analyze` 接口的小型**定性基准测试**。

## 图片列表

| 文件 | 期望情绪 | 来源 | 授权 |
|------|----------|--------|---------|
| `anger.jpg` | 愤怒 | FACES 公开预览集 (`116_m_m_a_a.jpg`) | 仅供研究使用，详见 [FACES 授权](https://faces.mpdl.mpg.de/imeji/collection/IXTdg721TwZwyZ8e) |
| `disgust.jpg` | 厌恶 | FACES 公开预览集 (`066_y_m_d_a.jpg`) | 仅供研究使用 |
| `fear.jpg` | 恐惧 | FACES 公开预览集 (`066_y_m_f_a.jpg`) | 仅供研究使用 |
| `happiness.jpg` | 快乐 | FACES 公开预览集 (`066_y_m_h_a.jpg`) | 仅供研究使用 |
| `neutral.jpg` | 中性 | FACES 公开预览集 (`066_y_m_n_a.jpg`) | 仅供研究使用 |
| `sadness.jpg` | 悲伤 | FACES 公开预览集 (`066_y_m_s_a.jpg`) | 仅供研究使用 |
| `surprise.jpg` | 惊讶 | Wikimedia Commons — "Wow" by Marcus Quigmire | CC BY-SA 2.0 |

## 仓库中包含的内容

- `surprise.jpg` 已根据 [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/) 授权提交到 Git。
- 六张 FACES 图片（`anger.jpg`、`disgust.jpg`、`fear.jpg`、`happiness.jpg`、`neutral.jpg`、`sadness.jpg`）**未包含**在本仓库中。它们来自 [FACES 公开预览集](https://faces.mpdl.mpg.de/imeji/collection/IXTdg721TwZwyZ8e)，仅用于研究方法论展示。如需在研究预览范围外使用或再分发，请在 FACES 官网注册并遵守其发布协议。
- 运行 `python benchmarks/download_images.py` 可下载 FACES 图片，并在需要时重新下载 `surprise.jpg`。

## 运行基准测试

```bash
python benchmarks/run_benchmark.py
```

## 最新结果（CPU，端口 8003）

| 情绪 | 预测结果 | 分数 | 结果 |
|---------|-----------|-------|--------|
| 愤怒 | Anger | 0.357 | ✅ |
| 厌恶 | Disgust | 0.593 | ✅ |
| 恐惧 | Fear | 0.501 | ✅ |
| 快乐 | Happiness | 0.240 | ✅ |
| 中性 | （全部较低，最高 Disgust=0.042） | — | ✅ |
| 悲伤 | Sadness | 0.417 | ✅ |
| 惊讶 | Surprise | 0.562 | ✅ |

中性情绪的判定标准为：6 种情绪分数均低于 0.1。
