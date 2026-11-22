# GPU 構成が既定。CPU のみのマシンでは docker-compose.cpu.yml が
# BASE_IMAGE=ubuntu:22.04 と TORCH_INDEX_URL=.../whl/cpu を渡す。
# CUDA イメージは python3 と CUDA ランタイムを載せているだけなので、
# 素の ubuntu でも同じ手順が通る（nemo-toolkit は CPU でも動く。ただし遅い）。
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchaudio --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir "nemo-toolkit[asr]==3.0.0"

WORKDIR /app
# These scripts are self-contained: the server imports its colocated nemotron_asr module only.
COPY src/maai_sds/asr/nemotron_asr.py /app/nemotron_asr.py
COPY src/maai_sds/asr/nemotron_asr_server.py /app/nemotron_asr_server.py
ENTRYPOINT ["python", "/app/nemotron_asr_server.py", "--host", "0.0.0.0"]
