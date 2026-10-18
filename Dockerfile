# MAHER wildlife identification web app.
# Build context must be the repo root (paths below reference project/src, project/frontend).
FROM python:3.11-slim

WORKDIR /app

COPY project/src/api/requirements.txt ./src/api/requirements.txt
RUN pip install --no-cache-dir -r ./src/api/requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# project/src and project/frontend must stay siblings under /app - api/app.py
# resolves the frontend directory relative to its own file location assuming
# that layout (same as in the real repo).
COPY project/src ./src
COPY project/frontend ./frontend

WORKDIR /app/src

EXPOSE 5000

CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "api.app:app"]
