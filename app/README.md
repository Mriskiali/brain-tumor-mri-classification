# app/

Flask web app untuk inferensi (yang di-deploy ke Railway).

Letakkan di sini:
- `app.py` — endpoint `/` (upload citra MRI) + inferensi model + tampilan hasil & indikasi anomali
- `templates/` — HTML template
- `Procfile` — `web: gunicorn app:app`
- `Procfile` opsional `railway.toml`/`nixpacks.toml` kalau ada config Railway spesifik

Catatan: `app.py` harus load model dari `../model/` atau dari env var `MODEL_PATH`.
