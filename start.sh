#!/bin/bash
uvicorn worker:app --host 127.0.0.1 --port 8000 &
n8n start
