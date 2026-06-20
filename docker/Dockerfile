FROM python:3.11-slim

WORKDIR /app

# noise package requires C compiler; fonts-nanum for Korean labels in matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc python3-dev fonts-nanum \
        libosmesa6-dev libgl1-mesa-dev libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "benchmark/run_benchmark.py"]
