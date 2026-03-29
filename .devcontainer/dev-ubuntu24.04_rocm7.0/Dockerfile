# Use ROCm build container on Ubuntu 24.04 as base
ARG ROCM_VERSION="7.0.2"
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete 
ENV ROCM_VERSION="${ROCM_VERSION}"

# Copy uv into image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Ensure system packages are up to date and install dependencies
RUN << 'EOF'
apt-get update
apt-get -y full-upgrade
apt-get install -y libglib2.0-0 \
                   git \
                   libegl1 \
                   libfontconfig1 \
                   libxkbcommon-x11-0 \
                   libdbus-1-3 \
                   libopengl0
apt-get clean && rm -rf /var/lib/apt/lists/*
EOF

# Set up Python virtual environment and upgrade pip and setuptools
USER ubuntu
WORKDIR /home/ubuntu

RUN curl -fsSL https://claude.ai/install.sh | bash

# Ensure the installed binary is on the `PATH`
ENV PATH="/home/ubuntu/.local/bin/:$PATH"

ENV VIRTUAL_ENV="/home/ubuntu/venv"
RUN uv venv $VIRTUAL_ENV --python 3.11
RUN uv pip install --no-cache-dir --upgrade pip setuptools wheel
RUN uv pip install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# This is a supposed fix to help with gfx1151
# See: https://github.com/pytorch/pytorch/issues/164346
ENV HSA_OVERRIDE_GFX_VERSION="11.0.0"
