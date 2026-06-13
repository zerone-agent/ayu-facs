FROM python:3.11-slim

WORKDIR /app

# 阿里云 apt 镜像源
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 配置 pip 全局清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 预装 torch 依赖（加速后续安装）
RUN pip install --no-cache-dir \
    'numpy==1.26.4' pillow sympy networkx jinja2 filelock fsspec 'typing-extensions>=4.8.0'

# 装 CPU 版 PyTorch（清华源找不到，从 PyTorch 官方 CPU 源拉）
RUN pip install --no-cache-dir \
    torch==2.8.0 torchvision==0.23 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 装 openface-test + FastAPI 等（openface-test 锁死所有版本，不会回溯）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 下载 OpenFace 3.0 模型权重（通过 hf-mirror 国内镜像，和 openface download 一致）
ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='nutPace/openface_weights', local_dir='./weights', repo_type='model')"

COPY config.py facs_engine.py server.py test_client.html ./

ENV OPENFACE_WEIGHTS_DIR=/app/weights
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
