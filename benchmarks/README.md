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
| `sadness.jpg` | sadness | FACES public preview (`066_y_m_s_a.jpg.jpg`) | Research-only |
| `surprise.jpg` | surprise | Wikimedia Commons — "The sincerity of amazement" by Alberto Castello | CC BY-SA 4.0 |

**Note:** The FACES images come from the [public preview collection](https://faces.mpdl.mpg.de/imeji/collection/IXTdg721TwZwyZ8e). They are provided for research methodology illustration. For broader use or redistribution, register at the FACES website and follow their release agreement.

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
| surprise | Surprise | 0.205 | ✅ |

Neutral is considered correct when all 6 emotion scores are below 0.1.
