#!/bin/sh
# Phone cameras only work on HTTPS pages, so make a throwaway self-signed certificate on first start.
mkdir -p /certs
[ -f /certs/cert.pem ] || openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=ocr-camera" \
  -keyout /certs/key.pem -out /certs/cert.pem 2>/dev/null
exec uvicorn webapp.reader_server:app --host 0.0.0.0 --port 8444 --ssl-keyfile /certs/key.pem --ssl-certfile /certs/cert.pem --no-access-log
