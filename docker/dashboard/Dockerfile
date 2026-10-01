FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements/base.txt /tmp/base.txt
COPY requirements/app.txt /tmp/app.txt
RUN pip install --no-cache-dir -r /tmp/app.txt

COPY src/ /app/src/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e .

COPY .streamlit/ /app/.streamlit/

EXPOSE 8501
CMD ["streamlit", "run", "src/netshield/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
