# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Build FastAPI Python Backend
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Copy python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Copy built frontend assets from Stage 1
COPY --from=frontend-builder /app/static ./static

# Train models on dataset subset
RUN python scripts/train_models.py || echo "Training skipped (data files may not be present)"

EXPOSE 8501

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8501"]
