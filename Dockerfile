FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pymupdf \
    opencv-python-headless \
    numpy \
    fastapi \
    uvicorn \
    pydantic \
    requests

COPY worker.py /app/worker.py

EXPOSE 8080

CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8080"]
