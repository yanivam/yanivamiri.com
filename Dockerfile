FROM python:3.11-slim

WORKDIR /app

COPY cogsci_demo/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cogsci_demo/ .

ENV DEMO_DATA_DIR=/data
ENV OVERRIDE_RNG_SEED=42

EXPOSE 8001

CMD ["sh", "-c", "uvicorn demo.api:app --host 0.0.0.0 --port ${PORT:-8001}"]
