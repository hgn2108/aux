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

# Weights go in the image: fetching them on first request would make every cold start a
# multi-minute wait, and Cloud Run's writable filesystem is memory-backed, so the download
# would cost RAM as well.
#
# This runs BEFORE the source is copied, and the script it runs imports nothing from the
# project, so that editing any file leaves the several-gigabyte layer untouched. With the
# copy above it, a one-line change re-downloaded all 3.6GB.
COPY --chown=user scripts/prefetch_models.py scripts/
RUN python scripts/prefetch_models.py

# Offline from here on, and only from here on -- the prefetch above needs the network.
#
# Having the weights in the image is not enough on its own. The Hugging Face libraries
# still call the hub on every load to check whether a cached model is current, and a Cloud
# Run container goes out through a shared egress address: those calls came back
# "429 Too Many Requests -- we had to rate limit your IP", and the retries were what made
# the first search appear to hang rather than fail.
#
# Offline mode makes the cache authoritative, so nothing leaves the container and the
# check costs nothing.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --chown=user . .
RUN pip install --no-cache-dir --user --no-deps -e .

# Cloud Run sets PORT and ignores EXPOSE; a Space uses app_port from its README. Binding to
# ${PORT:-7860} serves both from one image.
EXPOSE 7860
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-7860}"]
