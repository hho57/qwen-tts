#FROM rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.2
FROM rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1

ENV TRANSFORMERS_CACHE=/app/cache
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installation des dépendances système pour l'audio
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# On force la réinstallation de transformers depuis la source
RUN pip install --no-cache-dir flask accelerate librosa soundfile einops sentencepiece
RUN pip install --no-cache-dir --upgrade git+https://github.com/huggingface/transformers.git

COPY app/ /app/

CMD ["python", "/app/app.py"]


