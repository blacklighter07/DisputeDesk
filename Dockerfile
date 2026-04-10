FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
ENV ENV_README_PATH=/home/user/app/README.md

WORKDIR $HOME/app

COPY --chown=user server/requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --user --no-cache-dir -r /tmp/requirements.txt

COPY --chown=user . $HOME/app

RUN python -m pip install --user --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
