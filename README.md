# Customer Churn & Financial Retention Platform

![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient_boosting-2E7D32)
![MLflow](https://img.shields.io/badge/MLflow-tracking_%26_registry-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-inference_API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-ROI_simulator-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Sistema end-to-end de predicción de fuga de clientes para banca/retail, diseñado alrededor de una idea central: **un modelo de churn no vale por su AUC, vale por cuánto dinero deja de perder la empresa cuando se usa para decidir a quién contactar.**

## El resultado, primero

Evaluado en un holdout de test nunca visto durante entrenamiento ni optimización de umbral:

| Estrategia | Valor financiero neto | Clientes contactados |
|---|---:|---:|
| No contactar a nadie | −$809,055 | 0% |
| Contactar a todos (sin modelo) | +$42,717 | 100% |
| **Umbral óptimo del modelo** | **+$75,847** | 78.6% |

El modelo casi duplica el retorno de una campaña indiscriminada, contactando a menos gente. Ese es el producto: no "quién va a fugarse", sino "a quién conviene ofrecerle algo, y cuánto vale hacerlo".

## Arquitectura

```
┌────────────────────────┐
│  make_dataset.py        │  10,000 clientes sintéticos + LTV proyectado
└────────────┬─────────────┘
             │
             ▼
┌────────────────────────┐
│       train.py           │  LightGBM + split 50/30/20 + MLflow (tracking +
│  (MLflow: SQLite local)  │  registry) + búsqueda de umbral por retorno ($)
└────────────┬─────────────┘
             │
    ┌────────┴─────────┐
    │  data/processed/   │  churn_model.joblib + model_metadata.json
    └────────┬─────────┘
             │
   ┌─────────┴──────────┐
   ▼                     ▼
┌─────────────┐   ┌──────────────────┐
│  api/main.py │◄──┤ app/streamlit_app │  Simulador de ROI (recalcula el umbral
│  (FastAPI)   │   │  .py              │  en vivo) + predicción individual (vía API)
└─────────────┘   └──────────────────┘
       │                    │
       └─────────┬──────────┘
                  ▼
        docker-compose.yml
   (Dockerfile multietapa, un target por servicio)
```

## El proceso (las 3 fases)

Este proyecto se construyó en fases, entregadas y verificadas una por una — no se escribió todo de una vez y se asumió que funcionaba.

### Fase 1 — Datos y modelado

`src/data/make_dataset.py` genera 10,000 clientes bancarios sintéticos con un modelo de riesgo de fuga logístico (antigüedad, satisfacción, quejas, actividad, número de productos) y un **LTV proyectado** (ingreso mensual actual × meses de vida útil futura esperados, según actividad).

`src/models/train.py` entrena LightGBM sobre un split 50/30/20 (train / val / test), trackea todo en MLflow (SQLite local, con Model Registry funcionando), y — el corazón del proyecto — busca el **umbral de decisión que maximiza el retorno financiero neto** en validación, no el que maximiza F1 o accuracy.

**Esto no salió bien a la primera.** La primera versión del umbral óptimo era degenerado: 0.01, básicamente "contactar a todo el mundo". La razón, una vez diagnosticada: con la asimetría de costos de este problema (no contactar a un churner real cuesta su LTV completo; contactar a alguien que no lo era cuesta solo el precio de la campaña), el punto de indiferencia matemático es `costo / (LTV × (1 + tasa_de_éxito))` — y con un costo de campaña trivial frente al LTV en juego, ese punto cae por debajo de casi cualquier probabilidad predicha, así que "contactar a todos" gana matemáticamente sin que el modelo aporte nada. Ajustar solo el costo tampoco alcanzaba: subirlo demasiado volvía negativo el valor en *cualquier* umbral. La solución fue dos cambios combinados: un costo de campaña realista ($100, una oferta real — descuento de comisiones, bono — no una llamada trivial) y un modelo de riesgo con menos ruido (ROC-AUC 0.75 → 0.82), hasta que el óptimo cayó en una zona genuinamente interior y rentable. El detalle completo está comentado en `src/models/train.py`.

### Fase 2 — API y UI

`src/api/main.py`: FastAPI + Pydantic v2, con `lifespan` (no el `on_event` deprecado) para cargar el modelo al arrancar. Antes de escribirla verifiqué empíricamente un detalle no obvio de LightGBM en producción: hace falta castear las columnas categóricas a `dtype="category"` en cada request (si no, falla con `"train and valid dataset categorical_feature do not match"`), pero **no** hace falta replicar los niveles exactos de categorías vistos en entrenamiento — el booster guarda su propio mapeo interno (`pandas_categorical`) y lo aplica solo. Endpoints: `/health`, `/model/info`, `/predict`, `/predict/batch`.

`src/app/streamlit_app.py`: dos pestañas. **Simulador de ROI**, con sliders para costo de campaña y tasa de éxito que recalculan el umbral óptimo y el retorno en vivo sobre los 10,000 clientes. **Predicción individual**, que llama a la API real (no carga el modelo por su cuenta) — arquitectura de microservicios genuina.

### Fase 3 — Contenedorización y tests

`Dockerfile` multietapa: `builder` (compila con `build-essential`/`cmake`) → `runtime` (base mínima compartida, con `libgomp1` instalado — sin esa librería, LightGBM falla al importar en Debian slim con `libgomp.so.1: cannot open shared object file`, un gotcha real que vale la pena dejar documentado) → `api` y `app` como targets finales del mismo Dockerfile, para no duplicar la definición de dependencias. `docker-compose.yml` levanta ambos servicios, monta `data/` como volumen de solo lectura (los artefactos del modelo no se hornean en la imagen), y usa un healthcheck sin `curl` (vía `urllib` de la stdlib) para que `app` espere a que `api` esté realmente lista.

`tests/`: suite de integración con `pytest`. `conftest.py` genera el dataset y entrena el modelo automáticamente si no existen — verifiqué esto de verdad borrando todos los artefactos y corriendo `pytest tests/` desde cero (37s, 16/16 tests verdes), no solo lo asumí. `test_data_and_training.py` cubre el generador y la lógica financiera (incluye una prueba explícita de que el óptimo por retorno nunca es peor que un umbral fijo ingenuo). `test_api.py` prueba la API real con `TestClient`, incluyendo casos de validación de Pydantic (422 en geografía inválida, en score fuera de rango) y el flujo completo con LTV.

**Lo que no pude verificar en este entorno:** no hay Docker instalado en esta máquina, así que el `Dockerfile`/`docker-compose.yml` están escritos con cuidado pero no compilados ni ejecutados de punta a punta — revisalos antes de un despliegue real. Tampoco hay un navegador/`chromium-cli` disponible para una captura visual de Streamlit; lo validé indirectamente (arranque limpio sin excepciones, más una réplica independiente del cálculo exacto del simulador).

## Estructura del proyecto

```
customer-churn-mlops-platform/
├── data/
│   ├── raw/                        # customers.csv (generado, no versionado)
│   └── processed/                  # churn_model.joblib, model_metadata.json,
│                                    #   figures/threshold_vs_value.png (generados)
├── src/
│   ├── data/
│   │   └── make_dataset.py         # Genera 10,000 clientes sintéticos + LTV
│   ├── models/
│   │   └── train.py                # LightGBM + MLflow + umbral por retorno financiero
│   ├── api/
│   │   └── main.py                 # FastAPI: /predict, /predict/batch, /model/info
│   └── app/
│       └── streamlit_app.py        # Simulador de ROI + predicción individual
├── tests/
│   ├── conftest.py                 # Auto-bootstrap de dataset + modelo para CI
│   ├── test_data_and_training.py
│   └── test_api.py
├── Dockerfile                      # Multietapa: builder -> runtime -> api / app
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
# 1. Generar el dataset sintético
python -m src.data.make_dataset

# 2. Entrenar (LightGBM + MLflow + optimización de umbral financiero)
python -m src.models.train

# 3a. Levantar la API
uvicorn src.api.main:app --reload

# 3b. Levantar la UI (en otra terminal; espera que la API esté corriendo)
streamlit run src/app/streamlit_app.py

# Ver los experimentos trackeados en MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Pruebas de integración (auto-genera dataset/modelo si hace falta)
pytest tests/ -v

# Todo containerizado
docker compose up --build
# API en http://localhost:8000/docs -- UI en http://localhost:8501
```

## Referencia de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servicio y si el modelo está cargado |
| `GET` | `/model/info` | Metadata del modelo activo: umbral óptimo, AUC, supuestos de negocio |
| `POST` | `/predict` | Predicción para un cliente; si se envía `ltv`, agrega el valor financiero esperado |
| `POST` | `/predict/batch` | Predicción para una lista de clientes |

Ejemplo:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "credit_score": 550, "geography": "West", "gender": "Male", "age": 35,
  "tenure_years": 0, "balance": 500, "num_products": 1, "has_credit_card": false,
  "is_active_member": false, "estimated_salary": 40000, "num_complaints": 5,
  "satisfaction_score": 1, "monthly_fee_revenue": 25.0, "ltv": 450
}'
# {"churn_probability":0.9741,"optimal_threshold":0.03,"recommended_action":"contact","expected_financial_value":31.51}
```

## Stack técnico

| Herramienta | Rol |
|---|---|
| **LightGBM** | Modelo de clasificación de churn (categóricas nativas) |
| **MLflow** | Tracking de experimentos, métricas de negocio y Model Registry (SQLite local) |
| **FastAPI + Pydantic v2** | API de inferencia con validación de esquema estricta |
| **Streamlit** | UI del simulador de ROI y predicción individual |
| **Docker** | Imagen multietapa, un `target` por servicio |
| **pytest** | Suite de integración (datos, lógica financiera, API real) |
| **pandas / numpy / scikit-learn** | Preparación de datos y utilidades de modelado |

## Limitaciones conocidas

- Los datos son sintéticos (generador propio, sin conexión a un dataset bancario real) — el patrón de riesgo y el LTV son supuestos de diseño, documentados y calibrados, no observaciones.
- El umbral óptimo depende de `campaign_cost` y `retention_success_rate`, que son supuestos de negocio ajustables (por eso el simulador de ROI existe: para explorar qué pasa si cambian).
- La selección de umbral en un set de validación de tamaño moderado es sensible a la cola pesada del LTV — un puñado de clientes de alto valor puede mover el total en decenas de miles de dólares. Un sistema en producción debería usar validación cruzada o un holdout más grande para un umbral más estable.
