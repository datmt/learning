#!/bin/sh
# Export a trained checkpoint to ONNX and restart the test app on https://<this machine>:8444
# Usage: ./reload_model.sh [checkpoint.pt]     (default: the most recently saved data/crnn_final_*.pt)
cd "$(dirname "$0")" || exit 1
PY=/home/dat/jupyter-venv/bin/python
CK="${1:-$(ls -t data/crnn_final_*.pt 2>/dev/null | head -n 1)}"
[ -f "$CK" ] || { echo "no checkpoint found (pass one as the first argument)"; exit 1; }
echo "checkpoint: $CK  (saved $(date -r "$CK" '+%Y-%m-%d %H:%M'))"

PYTHONPATH=deploy/pylib "$PY" export_onnx.py "$CK" deploy/new.onnx 2>&1 | grep -v "Warning\|_generic_rnn" || true
[ -s deploy/new.onnx ] || { echo "export failed - the running model was NOT changed"; exit 1; }
[ -f deploy/crnn.onnx ] && cp deploy/crnn.onnx deploy/crnn.prev.onnx
mv deploy/new.onnx deploy/crnn.onnx; echo "$CK" > deploy/crnn.onnx.source

# stop the old server by PID (never pkill -f: it can match this script's own command line)
for P in $(ps -eo pid,args | awk '/uvicorn webapp.reader_server/ && !/awk/ {print $1}'); do kill "$P"; done
sleep 1
setsid nohup ./run_reader.sh > deploy/reader.log 2>&1 &
for i in 1 2 3 4 5 6 7 8 9 10; do
  [ "$(curl -sk -o /dev/null -w '%{http_code}' https://localhost:8444/)" = "200" ] && { echo "serving $CK at https://$(hostname -I | awk '{print $1}'):8444/  (reload the page on the phone)"; exit 0; }
  sleep 1
done
echo "server did not come up - see deploy/reader.log"; exit 1
