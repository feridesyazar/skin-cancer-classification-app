FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render builds the project from GitHub.
# The model is not stored on GitHub, so download it from Hugging Face.
# During a local Docker build, the existing local model is used instead.
RUN if [ ! -f /app/skin_cancer_model.keras ]; then \
        echo "Downloading model from Hugging Face..." && \
        curl -L --fail --retry 3 \
        -o /app/skin_cancer_model.keras \
        https://huggingface.co/ferides/skin-cancer-classification-model/resolve/main/skin_cancer_model.keras; \
    else \
        echo "Local model file found."; \
    fi

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 180 app:app"]
