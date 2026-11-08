#!/bin/bash
# Endless training: restarts the trainer whenever it exits, re-reading runs/r1/args.txt each time.
cd "$(dirname "$0")"

# Python of the local virtual environment (Windows or Unix layout), else whatever "python" is on PATH.
PY=python
for cand in .venv/Scripts/python.exe .venv/Scripts/python .venv/bin/python; do
  if [ -x "$cand" ]; then PY="$cand"; break; fi
done

while true; do
  "$PY" -m az.train $(cat runs/r1/args.txt) 2>&1 | grep -v -i warn
  echo "$(date +%H:%M:%S) trainer exited, restarting" | tee -a runs/r1/log.txt
  sleep 10
done
