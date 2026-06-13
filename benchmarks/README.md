# AYU FACS Benchmark Images

This directory contains a small qualitative benchmark for the `/analyze` endpoint.

## Images

| File | Expected | Source | License |
|------|----------|--------|---------|
| `anger.jpg` | anger | FACES public preview (`116_m_m_a_a.jpg`) | Research-only, see [FACES license](https://faces.mpdl.mpg.de/imeji/collection/IXTdg721TwZwyZ8e) |
| `disgust.jpg` | disgust | FACES public preview (`066_y_m_d_a.jpg`) | Research-only |
| `fear.jpg` | fear | FACES public preview (`066_y_m_f_a.jpg`) | Research-only |
| `happiness.jpg` | happiness | FACES public preview (`066_y_m_h_a.jpg`) | Research-only |
| `neutral.jpg` | neutral | FACES public preview (`066_y_m_n_a.jpg`) | Research-only |
| `sadness.jpg` | sadness | FACES public preview (`066_y_m_s_a.jpg`) | Research-only |
| `surprise.jpg` | surprise | Wikimedia Commons — "Wow" by Marcus Quigmire | CC BY-SA 2.0 |

## What's in this repository

- `surprise.jpg` is **committed** to Git under the [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/) license.
- The six FACES images (`anger.jpg`, `disgust.jpg`, `fear.jpg`, `happiness.jpg`, `neutral.jpg`, `sadness.jpg`) are **not included** in this repository. They come from the [FACES public preview collection](https://faces.mpdl.mpg.de/imeji/collection/IXTdg721TwZwyZ8e), which is provided for research methodology illustration only. To use or redistribute them beyond research preview, register at the FACES website and follow their release agreement.
- Run `python benchmarks/download_images.py` to download the FACES images and re-download `surprise.jpg` if needed.

## Run the benchmark

```bash
python benchmarks/run_benchmark.py
```

## Latest results (CPU, port 8003)

| Emotion | Predicted | Score | Result |
|---------|-----------|-------|--------|
| anger | Anger | 0.357 | ✅ |
| disgust | Disgust | 0.593 | ✅ |
| fear | Fear | 0.501 | ✅ |
| happiness | Happiness | 0.240 | ✅ |
| neutral | (all low, max Disgust=0.042) | — | ✅ |
| sadness | Sadness | 0.417 | ✅ |
| surprise | Surprise | 0.562 | ✅ |

Neutral is considered correct when all 6 emotion scores are below 0.1.
