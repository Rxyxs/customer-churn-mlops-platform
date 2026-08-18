# syntax=docker/dockerfile:1

# ---- Stage 1: builder -- instala dependencias con las herramientas de compilacion
# que LightGBM/XGBoost necesitan, sin llevarlas a la imagen final. ----
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Stage 2: runtime -- base minima compartida por los servicios API y UI.
# libgomp1 es el gotcha clasico de LightGBM en Debian slim: sin esa lib comparte
# falla al importar con "libgomp.so.1: cannot open shared object file". ----
FROM python:3.11-slim AS runtime
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY src/ ./src/

# El modelo entrenado (data/processed/) se monta como volumen en docker-compose,
# no se hornea en la imagen -- separa artefactos de modelo del código de la app.

# ---- Stage 3: api -- servicio FastAPI de inferencia. ----
FROM runtime AS api
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- Stage 4: app -- servicio Streamlit (UI + simulador de ROI). ----
FROM runtime AS app
EXPOSE 8501
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
