# Streamlit is no longer a first-class Hugging Face Spaces SDK, so a hosted demo runs as a
# container. This image targets Cloud Run and works anywhere else that takes one.
FROM python:3.12-slim

# libsndfile is needed by soundfile, which MuQ pulls in through librosa. PyAV ships
# manylinux wheels with the ffmpeg libraries bundled, so no ffmpeg packages are required --
# and the ffmpeg *binary* is deliberately absent: this project decodes through PyAV and
# hands Whisper a decoded array rather than shelling out.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Containers on Cloud Run and Spaces both run unprivileged with a writable home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    AUX_PUBLIC=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /home/user/app

# torch and torchaudio come from the CPU index explicitly, before anything that depends on
# them. The default PyPI wheels carry ~2.5GB of CUDA libraries a CPU instance cannot use,
# and --extra-index-url alone does not prevent them: pip would still be free to resolve the
# newer CUDA build. Installing them first means muq finds its requirement satisfied.
RUN pip install --no-cache-dir --user \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2" torchaudio

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
RUN pip install --no-cache-dir --user --no-deps -e .

# Weights go in the image. Fetching them on first request would make every cold start a
# multi-minute wait, and the writable filesystem on Cloud Run is memory-backed, so the
# download would cost RAM as well as latency.
RUN python scripts/prefetch_models.py

# Cloud Run sets PORT and ignores EXPOSE; a Space uses app_port from its README. Binding to
# ${PORT:-7860} serves both from one image.
EXPOSE 7860
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-7860}"]
