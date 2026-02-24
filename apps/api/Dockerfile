FROM python:3.11-slim

WORKDIR /app

# Install build deps for psycopg binary (mostly not needed, but keep minimal)
RUN pip install --no-cache-dir -U pip

COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir -e .

COPY src /app/src
COPY apps/__init__.py /app/apps/__init__.py
COPY apps/api /app/apps/api
COPY tests /app/tests

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
