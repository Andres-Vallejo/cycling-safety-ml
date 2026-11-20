FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-core.lock.txt ./
RUN python -m pip install --no-cache-dir -r requirements-core.lock.txt
COPY pyproject.toml LICENSE README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps . \
    && useradd --uid 10001 --create-home appuser
COPY artifacts/model.json ./artifacts/model.json
USER appuser
EXPOSE 8000
CMD ["uvicorn", "cycling_safety.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
