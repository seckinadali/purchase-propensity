# Dockerfile for the FastAPI serving layer.
#
# Build:  docker build -t purchase-propensity-serve .
# Run:    docker run -p 8000:8000 purchase-propensity-serve
#
# Packages the serving path only — model + serve.py + config + runtime deps.

FROM python:3.11-slim

# libgomp1 is the GNU OpenMP runtime; LightGBM and XGBoost dlopen it for parallel
# tree ops. The slim base image strips it, so the wheel installs cleanly but the
# server crashes on first import. Adding it costs ~50KB in the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv from the official image is faster and cache-friendlier than pip install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Lockfile-first so dependency layers only invalidate when deps change, not on
# every code edit. --no-install-project skips the (empty) project package itself.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code + model artifacts. Only the files serve.py actually needs.
COPY serve.py ./
COPY src/config.py ./src/config.py
COPY models/lgb_model.pkl models/model_info.json ./models/

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
