#!/bin/sh
cd "$(dirname "$0")"
exec .venv/bin/uvicorn webapp.reader_server:app --host 0.0.0.0 --port 8444 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem --no-access-log
