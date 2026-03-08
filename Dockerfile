FROM rocm/pytorch

# On définit les variables pour l'APU au niveau système
ENV HSA_OVERRIDE_GFX_VERSION=11.5.0
ENV ROCM_PATH=/opt/rocm
ENV PATH="$ROCM_PATH/bin:$ROCM_PATH/opencl/bin:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg git libsndfile1 && rm -rf /var/lib/apt/lists/*

# Force la réinstallation des libs et de transformers (support qwen3_tts)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir flask accelerate librosa soundfile einops sentencepiece && \
    pip install --no-cache-dir --force-reinstall git+https://github.com/huggingface/transformers.git

COPY app/ /app/

CMD ["python", "/app/app.py"]

