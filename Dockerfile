FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETPLATFORM

RUN set -eux; \
    apt-get update; \
    apt-get install --no-install-recommends -y curl ffmpeg ca-certificates; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    if [ "${TARGETPLATFORM}" != "linux/amd64" ]; then \
        apt-get update; \
        apt-get install --no-install-recommends -y build-essential; \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
        rm -rf /var/lib/apt/lists/*; \
    fi
ENV PATH="/root/.cargo/bin:${PATH}"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN useradd -m -u 10001 appuser

WORKDIR /app
RUN mkdir -p voices config && chown -R appuser:appuser /app

ARG USE_ROCM=0
ENV USE_ROCM=${USE_ROCM}

COPY requirements*.txt constraints*.txt /app/
RUN set -eux; \
    pip install -U pip; \
    if [ "${USE_ROCM}" = "1" ] && [ -f requirements-rocm.txt ]; then mv requirements-rocm.txt requirements.txt; fi; \
    if [ -f constraints.txt ]; then \
        pip install --no-cache-dir -r requirements.txt -c constraints.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

COPY *.py *.sh *.default.yaml README.md LICENSE /app/
RUN chown -R appuser:appuser /app

ARG PRELOAD_MODEL
ENV PRELOAD_MODEL=${PRELOAD_MODEL}
ENV TTS_HOME=voices
ENV HF_HOME=voices
ENV COQUI_TOS_AGREED=1

USER appuser

CMD ["bash", "startup.sh"]
