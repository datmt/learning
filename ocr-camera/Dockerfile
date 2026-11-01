FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends openssl libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ocr_model.py crop.py ./
COPY webapp ./webapp
COPY model ./model
COPY scripts/docker-entry.sh /entry.sh
ENV OCR_ONNX=/app/model/crnn.onnx OCR_READER_LOG=/logs
VOLUME /logs
EXPOSE 8444
CMD ["sh", "/entry.sh"]
