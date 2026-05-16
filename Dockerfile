FROM python:3.12-slim

RUN pip install uv

WORKDIR /workspace

COPY pyproject.toml .python-version ./
RUN uv sync --no-install-project

COPY src/ ./src/
COPY conf/ ./conf/

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "src/poc/app/main.py", "--server.address=0.0.0.0"]
