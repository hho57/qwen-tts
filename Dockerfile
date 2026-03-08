FROM rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.2

ENV TRANSFORMERS_CACHE=/app/cache
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0

WORKDIR /app

COPY app/ /app/

RUN pip install --no-cache-dir \
    flask \
    transformers \
    accelerate \
    librosa \
    soundfile

CMD ["python", "/app/app.py"]