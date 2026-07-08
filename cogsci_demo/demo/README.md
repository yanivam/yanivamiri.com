# Booth QR Demo

Interactive web app for the poster: visitors pick two drugs and three attention factors, then compare their policy against the Learner Agent on the same 6-drug scenario.

## Run locally

```bash
pip install -r requirements.txt
python demo/run_server.py
```

Open **http://localhost:8001**

## Deploy online (recommended for the poster)

Hosting online gives you a **stable URL** for the poster QR code and keeps share links (`/r/{session_id}`) working after the conference.

### Railway (easiest)

1. Push this repo to GitHub (include the `demo/` folder and `demo/data/attention_weights/`).
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Add a **Volume** to the service:
   - Mount path: `/data`
4. Add a variable (if not already set by the Dockerfile):
   - `DEMO_DATA_DIR` = `/data`
5. Deploy. Railway assigns a public URL like `https://your-app.up.railway.app`.
6. Optional: **Settings → Networking → Generate Domain** or attach a custom domain.

**Poster QR code** → encode that URL (homepage only, not a session link):

```
https://your-app.up.railway.app/
```

Scan it on your phone before printing the poster.

### Export data after the conference

Sessions and emails live in SQLite on the volume. Download `booth.db` from Railway’s volume UI, place it at `demo/data/booth.db`, then:

```bash
python demo/export_emails.py
python demo/export_choices.py
```

### Other hosts

The root `Dockerfile` works on Fly.io, Render, Google Cloud Run, etc. Set `DEMO_DATA_DIR` to a persistent disk mount so sessions survive redeploys.

```bash
docker build -t booth-demo .
docker run -p 8001:8001 -v booth-data:/data booth-demo
```

## What it does

1. **Select**: pick 2 drugs + 3 attention factors
2. **Email**: optional; stored with consent
3. **Simulate**: 5-week run: your policy vs learner
4. **Results**: attention comparison, focus timelines, outcomes

## Regenerate learner weights (optional)

```bash
python demo/precompute.py
```

Overwrites `demo/data/attention_weights/learned_attention_weights.json`.

## Poster copy

> Scan to run your attention policy  
> Compare your focus set with the Learner Agent on the same shortage scenario.
