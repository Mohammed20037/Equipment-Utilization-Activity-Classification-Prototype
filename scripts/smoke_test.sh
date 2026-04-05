#!/usr/bin/env bash
set -euo pipefail

cp -n .env.example .env || true

echo "[1/4] Starting stack..."
docker compose up --build -d

echo "[2/4] Waiting for services..."
sleep 12

echo "[3/4] Service status"
docker compose ps

echo "[4/4] Recent logs"
docker compose logs --tail=80 cv_service analytics_service ui_service

echo "Smoke test completed. Dashboard: http://localhost:8501"
