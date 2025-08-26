# Multi-stage Docker build for Safe RL production deployment
# Base image with CUDA support for GPU acceleration

# ==============================================================================
# Stage 1: Build stage
# ==============================================================================
FROM nvidia/cuda:11.8-devel-ubuntu20.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install virtualenv
RUN virtualenv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-prod.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements-prod.txt

# Copy source code and install package
COPY safe_rl_human_robot/ /app/safe_rl_human_robot/
COPY setup.py README.md LICENSE /app/
WORKDIR /app
RUN pip install -e .

# ==============================================================================
# Stage 2: Runtime stage
# ==============================================================================
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 AS runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-distutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --from=builder /app /app
WORKDIR /app

# Create non-root user for security
RUN groupadd -r saferl && useradd -r -g saferl -d /app -s /sbin/nologin saferl
RUN chown -R saferl:saferl /app
USER saferl

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "safe_rl_human_robot.src.deployment.api_server", "--host", "0.0.0.0", "--port", "8000"]