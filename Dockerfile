# Streamlit is no longer a first-class Spaces SDK, so a hosted demo runs as a Docker Space.
# This image also runs anywhere else that takes a container.
FROM python:3.12-slim

# PyAV needs the ffmpeg libraries (not the CLI -- this project decodes through PyAV and
# never shells out), and git is needed for the pip installs that resolve from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libswscale-dev libswresample-dev libavfilter-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as uid 1000 and mount a writable home there; anything written elsewhere at
# runtime fails. The model cache is several GB, so it has to land inside it.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    AUX_PUBLIC=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /home/user/app

# CPU-only torch: the default wheel pulls ~2.5GB of CUDA libraries that a CPU Space cannot
# use, and the image will not build within the size limit with them.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY --chown=user . .
RUN pip install --no-cache-dir --user --no-deps -e .

EXPOSE 7860
CMD ["streamlit", "run", "app.py"]
