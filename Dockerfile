#FROM rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.2
FROM rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1

ENV TRANSFORMERS_CACHE=/app/cache
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg git libsndfile1 && rm -rf /var/lib/apt/lists/*

# Force l'installation de la version de dev pour supporter qwen3_tts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir flask accelerate librosa soundfile einops sentencepiece && \
    pip install --no-cache-dir --force-reinstall git+https://github.com/huggingface/transformers.git

COPY app/ /app/

# On s'assure que le port correspond à ton compose
CMD ["python", "/app/app.py"]
