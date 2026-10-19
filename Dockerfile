FROM node:20-bookworm-slim

USER root

# Install Python, system graphics libraries for OpenCV, and curl
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install n8n globally
RUN npm install -g n8n

# Install Python vision dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    pymupdf \
    opencv-python-headless \
    numpy \
    fastapi \
    uvicorn \
    pydantic

# Setup directory and permissions
WORKDIR /app
COPY worker.py /app/worker.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh && \
    mkdir -p /home/node/.n8n && \
    chown -R 1000:1000 /home/node /app

USER 1000

ENV N8N_PROTOCOL=https
ENV HOME=/home/node

EXPOSE 7860

CMD ["/app/start.sh"]
