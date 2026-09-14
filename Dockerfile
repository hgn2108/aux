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

# All three torch packages come from the CPU index, in one resolve, before anything that
# depends on them.
#
# One index, because the three ship matched C++ extensions: torchvision built against a
# different torch registers no operators, and the first symbol MuQ reaches through x_clip
# fails with "operator torchvision::nms does not exist".
#
# torchvision is listed even though nothing here does vision. MuQ imports x_clip, which
# imports its visual-SSL module at package level, which imports torchvision -- so it is a
# hard requirement of an import chain rather than of any code path that runs.
#
# CPU index, because the default PyPI wheels carry ~2.5GB of CUDA libraries a CPU instance
# cannot use. --extra-index-url is not enough on its own: pip stays free to resolve the
# newer CUDA build from PyPI.
RUN pip install --no-cache-dir --user \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2" torchvision torchaudio

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Fail here, not eight minutes later in the model prefetch. This is the exact import chain
# that breaks when the torch packages come from different builds: muq -> x_clip ->
# torchvision, and a mismatched torchvision registers no operators.
RUN python -c "import torch, torchvision, x_clip; \
    print('torch', torch.__version__, '| torchvision', torchvision.__version__); \
    torchvision.ops.nms(torch.zeros(0, 4), torch.zeros(0), 0.5)"

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
