FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
COPY src/ /app/src/
COPY README.md .
COPY LICENSE .

RUN uv pip install --system '.[web]'

EXPOSE 8000

CMD ["promptmask-web"]