# yanivamiri.com

Personal site repo. The **CogSci 2026 attention demo** lives in [`cogsci_demo/`](cogsci_demo/).

## Demo (local)

```bash
cd cogsci_demo
pip install -r requirements.txt
python demo/run_server.py
```

Open http://localhost:8001

## Deploy the demo (conference hosting)

The demo stores sessions and emails in SQLite. Use a host with a **persistent volume** so data survives redeploys and share links (`/r/{session_id}`) keep working.

### Railway (recommended)

1. Push this repo to GitHub (already connected to `origin`).
2. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo** → select this repo.
3. Railway reads [`railway.toml`](railway.toml) and builds from the root [`Dockerfile`](Dockerfile).
4. Add a **Volume** to the service:
   - Mount path: `/data`
5. Add environment variables:
   - `DEMO_DATA_DIR` = `/data`
   - `DEMO_ADMIN_TOKEN` = a long random password (for `/admin` CSV downloads)
6. **Settings → Networking → Generate Domain** (e.g. `your-demo.up.railway.app`).
7. Optional: attach a custom domain such as `demo.yanivamiri.com` (CNAME to Railway).

**QR code / Squarespace link** → point to the public URL root (`/`), not a session link.

### Retrieve responses after the conference

Open **https://your-app.up.railway.app/admin**, enter your `DEMO_ADMIN_TOKEN`, then download:

- **contacts CSV** — names and emails (with consent)
- **choices CSV** — every run’s drug/factor picks

Or download `booth.db` from the Railway volume and export locally (see below).

Download `booth.db` from the Railway volume, place at `cogsci_demo/demo/data/booth.db`, then:

```bash
cd cogsci_demo
python demo/export_emails.py
python demo/export_choices.py
```
