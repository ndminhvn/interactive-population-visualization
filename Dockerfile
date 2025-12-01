FROM python:3.13-slim

WORKDIR /app

# Install VTK runtime dependencies including OSMesa for offscreen rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgomp1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libosmesa6 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8080

CMD ["python", "src/app.py", "--host", "0.0.0.0", "--port", "8080"]
