# NilaCare Backend VPS

Folder ini dibuat flat dan simpel seperti backend lama, supaya gampang di-remote ke VPS.

## Struktur

```text
backend-vps/
├─ __init__.py
├─ .dockerignore
├─ docker-compose.yml
├─ Dockerfile
├─ labels.py
├─ main.py
├─ MobileNetV2_best.h5
└─ requirements.txt
```

## Jalankan Dengan Docker Compose

```bash
docker compose up -d --build
```

Cek backend:

```bash
curl http://127.0.0.1:8091/health
```

## Jalankan Manual

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8091
```

## Environment

Kalau perlu atur CORS:

```env
CORS_ORIGINS=https://domain-frontend.com,http://localhost:3000
```

Di frontend, arahkan API ke backend VPS:

```env
NEXT_PUBLIC_API_BASE_URL=https://domain-backend.com
```
