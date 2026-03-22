#!/bin/bash
# Uso: ./deploy.sh root@12.34.56.78
set -e

REMOTE="${1:?Uso: ./deploy.sh usuario@ip}"
REMOTE_DIR="/opt/rag-buconos"

echo "▶ Subiendo archivos a $REMOTE:$REMOTE_DIR …"
ssh "$REMOTE" "mkdir -p $REMOTE_DIR/RAG_buconos"

rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.cache' \
    "$(dirname "$0")/"    "$REMOTE:$REMOTE_DIR/rag/"
rsync -avz \
    "$(dirname "$0")/../RAG_buconos/" "$REMOTE:$REMOTE_DIR/RAG_buconos/"

echo "▶ Instalando Docker si no está …"
ssh "$REMOTE" "command -v docker || (curl -fsSL https://get.docker.com | sh)"
ssh "$REMOTE" "command -v docker compose || apt-get install -y docker-compose-plugin"

echo "▶ Arrancando servicio …"
ssh "$REMOTE" "cd $REMOTE_DIR/rag && docker compose up -d --build"

echo ""
echo "✅ Listo → http://$(echo $REMOTE | cut -d@ -f2):8501"
