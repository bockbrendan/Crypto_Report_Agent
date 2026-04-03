import modal

# ── Image ─────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libpango-1.0-0", "libpangoft2-1.0-0", "libgdk-pixbuf2.0-0", "libffi-dev", "shared-mime-info")
    .pip_install_from_requirements("requirements.txt")
    .env({"FRONTEND_DIR": "/app/frontend"})
    .add_local_python_source("backend")
    .add_local_dir("frontend", remote_path="/app/frontend")
)

app = modal.App("crypto-report-agent", image=image)

# ── ASGI endpoint ─────────────────────────────────────────────────────────────

@app.function(
    secrets=[modal.Secret.from_name("crypto-intel-secrets")],
    timeout=300,      # pipeline takes 60–120s; 300s gives headroom
    min_containers=1, # cold-start mode (free tier) — set to 1 for always-on (~$1.40/mo)
)
@modal.asgi_app()
def web():
    from backend.agents.main import app as fastapi_app
    return fastapi_app
