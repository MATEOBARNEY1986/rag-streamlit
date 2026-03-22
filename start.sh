#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "✏️  Edita .env con tus claves y vuelve a ejecutar este script."
  exit 1
fi

if ! python3 -c "import streamlit" 2>/dev/null; then
  echo "📦 Instalando dependencias…"
  pip install -r requirements.txt
fi

streamlit run app.py --server.port 8501 --server.headless true
