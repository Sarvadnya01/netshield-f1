FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements/base.txt /tmp/base.txt
COPY requirements/app.txt /tmp/app.txt
RUN pip install --no-cache-dir -r /tmp/app.txt

COPY src/ /app/src/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "netshield.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
