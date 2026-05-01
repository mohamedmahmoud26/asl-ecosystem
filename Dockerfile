FROM python:3.10-slim

# system libs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD sh -c "uvicorn inference_api.fastdemo:app --host 0.0.0.0 --port ${PORT:-8000}"