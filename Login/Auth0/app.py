# app.py — Single-window Auth0 + Management API persistence (user_metadata.display_name)
# Requires: Flask, python-dotenv, requests, pywebview, authlib, keyring

import os, threading, tempfile, time, json
from pathlib import Path
from urllib.parse import urlencode, urlparse
from flask import Flask, request, jsonify, redirect, make_response, send_file
from dotenv import load_dotenv
import requests
import webview

from auth_flow import (
    Auth0Config, AuthState, make_pkce,
    exchange_code_for_tokens, refresh_tokens
)

# ---------- Fresh Edge WebView2 profile each run (avoid stale cache) ----------
USER_DATA_DIR = tempfile.mkdtemp(prefix="straightup-wv2-")
os.environ["WEBVIEW2_USER_DATA_FOLDER"] = USER_DATA_DIR

# ---------- Config ----------
APP_DIR = Path(__file__).resolve().parent
UI_DIR  = APP_DIR.parent / "UI"
ENV_PATH = APP_DIR / ".env"
load_dotenv(ENV_PATH)

PORT         = int(os.getenv("REDIRECT_PORT", "8765"))
DOMAIN_RAW   = (os.getenv("AUTH0_DOMAIN") or "").strip().rstrip("/")
CLIENT_ID    = (os.getenv("AUTH0_CLIENT_ID") or "").strip()
AUDIENCE     = (os.getenv("AUTH0_AUDIENCE") or "").strip() or None
REDIRECT_URI = f"http://127.0.0.1:{PORT}/callback"

MGMT_CLIENT_ID     = (os.getenv("MGMT_CLIENT_ID") or "").strip()
MGMT_CLIENT_SECRET = (os.getenv("MGMT_CLIENT_SECRET") or "").strip()
MGMT_AUDIENCE      = (os.getenv("MGMT_AUDIENCE") or "").strip()  # e.g. https://YOUR_DOMAIN/api/v2/

def _norm_domain(d: str) -> str:
    if not d: return ""
    d = d.rstrip("/")
    return d if d.startswith("http") else f"https://{d}"

DOMAIN = _norm_domain(DOMAIN_RAW)
u = urlparse(DOMAIN)
if not (u.scheme and u.netloc):
    raise RuntimeError("AUTH0_DOMAIN missing/invalid in .env (e.g., dev-xyz.us.auth0.com)")
if not CLIENT_ID:
    raise RuntimeError("AUTH0_CLIENT_ID missing in .env")

cfg   = Auth0Config(domain=DOMAIN, client_id=CLIENT_ID, redirect_uri=REDIRECT_URI, audience=AUDIENCE)
state = AuthState()

INDEX_URL = f"http://127.0.0.1:{PORT}/ui/index"
HOME_URL  = f"http://127.0.0.1:{PORT}/ui/home"

print("DEBUG AUTH0:", {"domain": cfg.domain, "client_id": cfg.client_id, "redirect": cfg.redirect_uri})
print("DEBUG WEBVIEW2 USER DATA:", USER_DATA_DIR)

# ---------- Flask ----------
app = Flask(__name__)

@app.get("/ui/index")
def ui_index():
    return send_file(UI_DIR / "index.html")

@app.get("/ui/home")
def ui_home():
    return send_file(UI_DIR / "home.html")

@app.get("/ui/dashboard")
def ui_dashboard():
    p = UI_DIR / "dashboard.html"
    return send_file(p) if p.exists() else ("<h1>Make UI/dashboard.html</h1>", 200)

@app.get("/favicon.ico")
def favicon():
    return ("", 204)

@app.get("/status")
def status():
    return jsonify({"logged_in": state.logged_in, "profile": state.profile if state.logged_in else None})

# ---------- Management API helpers (token cache + user read/patch) ----------
_mgmt_token = None
_mgmt_exp   = 0

def get_mgmt_token():
    """Client credentials grant for Auth0 Management API (M2M)."""
    global _mgmt_token, _mgmt_exp
    if _mgmt_token and time.time() < _mgmt_exp - 30:
        return _mgmt_token

    aud = MGMT_AUDIENCE or f"{DOMAIN}/api/v2/"
    payload = {
        "grant_type": "client_credentials",
        "client_id": MGMT_CLIENT_ID,
        "client_secret": MGMT_CLIENT_SECRET,
        "audience": aud
    }
    resp = requests.post(f"{DOMAIN}/oauth/token", json=payload, timeout=10)
    resp.raise_for_status()
    tok = resp.json()
    _mgmt_token = tok["access_token"]
    _mgmt_exp   = time.time() + int(tok.get("expires_in", 3600))
    return _mgmt_token

def mgmt_get_user(user_id: str):
    t = get_mgmt_token()
    url = f"{DOMAIN}/api/v2/users/{requests.utils.quote(user_id, safe='')}"
    r = requests.get(url, headers={"Authorization": f"Bearer {t}"}, timeout=10)
    r.raise_for_status()
    return r.json()

def mgmt_patch_user(user_id: str, user_metadata: dict):
    t = get_mgmt_token()
    url = f"{DOMAIN}/api/v2/users/{requests.utils.quote(user_id, safe='')}"
    body = {"user_metadata": user_metadata}
    r = requests.patch(url, headers={
        "Authorization": f"Bearer {t}",
        "Content-Type": "application/json"
    }, json=body, timeout=10)
    r.raise_for_status()
    return r.json()

# ---------- Profile API (persists display_name to user_metadata) ----------
@app.get("/profile")
def get_profile():
    return jsonify({"logged_in": state.logged_in, "profile": state.profile if state.logged_in else None})

@app.post("/profile")
def update_profile():
    if not state.logged_in:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()

    # Update local state for instant UI feedback
    if name:
        state.profile = dict(state.profile or {}, name=name)

    # Persist to Auth0 user_metadata
    try:
        sub = (state.profile or {}).get("sub")
        if sub and name and MGMT_CLIENT_ID and MGMT_CLIENT_SECRET:
            mgmt_patch_user(sub, {"display_name": name})
    except Exception as e:
        print("WARN: failed to persist display_name:", e)
        # still return ok, since local state is updated
    return jsonify({"ok": True, "profile": state.profile})

# ---------- Auth helpers ----------
def build_authorize_url(*, signup=False, login_hint=None):
    state._verifier, challenge = make_pkce()
    import secrets as _s; state._state = _s.token_urlsafe(16)
    params = {
        "client_id": cfg.client_id,
        "response_type": "code",
        "redirect_uri": cfg.redirect_uri,
        "scope": "openid profile email offline_access",
        "state": state._state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if AUDIENCE:
        params["audience"] = AUDIENCE
    if signup:
        params["screen_hint"] = "signup"
    if login_hint:
        params["login_hint"] = login_hint
    url = f"{cfg.domain}/authorize?{urlencode(params)}"
    print("DEBUG authorize URL:", url)
    return url

# Use JS redirect page (stable in embedded Chromium)
@app.get("/auth/login")
def auth_login():
    url = build_authorize_url(signup=False)
    return f"<!doctype html><meta charset='utf-8'><script>location.replace({url!r})</script>"

@app.get("/auth/signup")
def auth_signup():
    url = build_authorize_url(signup=True)
    return f"<!doctype html><meta charset='utf-8'><script>location.replace({url!r})</script>"

# OAuth callback -> after-auth -> home (merge metadata here)
@app.get("/callback")
def callback():
    code     = request.args.get("code")
    rx_state = request.args.get("state")
    try:
        # exchange -> fills state.profile (name/email/sub) and tokens
        exchange_code_for_tokens(cfg, state, code, rx_state)

        # enrich from Management API user_metadata
        try:
            sub = (state.profile or {}).get("sub")
            if sub and MGMT_CLIENT_ID and MGMT_CLIENT_SECRET:
                u = mgmt_get_user(sub)
                meta = u.get("user_metadata") or {}
                display_name = meta.get("display_name")
                if display_name:
                    state.profile = dict(state.profile or {}, name=display_name, email=u.get("email") or (state.profile or {}).get("email"))
        except Exception as e:
            print("WARN: could not fetch user_metadata:", e)

        return redirect(f"http://127.0.0.1:{PORT}/after-auth")
    except Exception as e:
        print("ERROR in /callback:", e)
        return f"Login error: {e}", 400

@app.get("/after-auth")
def after_auth():
    html = f"""<!doctype html><meta charset="utf-8">
<script>
try {{ window.location.replace({HOME_URL!r}); }}
catch(e) {{ window.location = {HOME_URL!r}; }}
</script>"""
    resp = make_response(html); resp.headers["Cache-Control"] = "no-store"; return resp

@app.get("/logout")
def logout():
    state.logged_in = False
    state.profile = {}
    state.access_token = None
    state.id_token = None
    state.clear_refresh()
    html = f"""<!doctype html><meta charset="utf-8">
<script>
try {{ window.location.replace({INDEX_URL!r}); }}
catch(e) {{ window.location = {INDEX_URL!r}; }}
</script>"""
    resp = make_response(html); resp.headers["Cache-Control"] = "no-store"; return resp

def run_flask():
    app.run(host="127.0.0.1", port=PORT, debug=False)

# ---------- Main ----------
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    MAIN_WIN = webview.create_window(
        title="Straight Up",
        url=f"http://127.0.0.1:{PORT}/ui/index",
        width=520, height=700,
        resizable=True, min_size=(420, 560),
    )
    webview.start(gui='edgechromium', debug=True)
