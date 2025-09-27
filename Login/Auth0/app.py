import os, threading
from pathlib import Path
from urllib.parse import urlencode, urlparse
from flask import Flask, request, jsonify, redirect
from dotenv import load_dotenv
import webview

from auth_flow import (
    Auth0Config, AuthState, make_pkce,
    exchange_code_for_tokens, refresh_tokens, build_logout_url
)

# --- load env from THIS folder ---
APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"
load_dotenv(ENV_PATH)

PORT         = int(os.getenv("REDIRECT_PORT", "8765"))
DOMAIN_RAW   = (os.getenv("AUTH0_DOMAIN") or "").strip().rstrip("/")
CLIENT_ID    = (os.getenv("AUTH0_CLIENT_ID") or "").strip()
AUDIENCE     = (os.getenv("AUTH0_AUDIENCE") or "").strip() or None
REDIRECT_URI = f"http://127.0.0.1:{PORT}/callback"

def _norm_domain(d: str) -> str:
    if not d: return ""
    d = d.rstrip("/")
    return d if d.startswith("http") else f"https://{d}"

DOMAIN = _norm_domain(DOMAIN_RAW)
u = urlparse(DOMAIN)
if not (u.scheme and u.netloc): raise RuntimeError("AUTH0_DOMAIN missing/invalid in .env")
if not CLIENT_ID: raise RuntimeError("AUTH0_CLIENT_ID missing in .env")

cfg   = Auth0Config(domain=DOMAIN, client_id=CLIENT_ID, redirect_uri=REDIRECT_URI, audience=AUDIENCE)
state = AuthState()

print("DEBUG AUTH0:", {"domain": cfg.domain, "client_id": cfg.client_id, "redirect": cfg.redirect_uri})

# --- Flask ---
app = Flask(__name__)

@app.get("/status")
def status():
    return jsonify({"logged_in": state.logged_in, "profile": state.profile if state.logged_in else None})

@app.get("/callback")
def callback():
    code     = request.args.get("code")
    rx_state = request.args.get("state")
    try:
        exchange_code_for_tokens(cfg, state, code, rx_state)
        return redirect(f"http://127.0.0.1:{PORT}/login-complete")
    except Exception as e:
        return f"Login error: {e}", 400

@app.get("/login-complete")
def login_complete():
    return """
    <script>
      if (window.pywebview && window.pywebview.api && window.pywebview.api.close_auth) {
        window.pywebview.api.close_auth();
      } else { window.close(); }
    </script>
    Logged in. You can close this tab.
    """

@app.get("/logout")
def logout():
    state.logged_in = False; state.profile = {}; state.access_token = None; state.id_token = None; state.clear_refresh()
    return redirect(build_logout_url(cfg))

def run_flask():
    app.run(host="127.0.0.1", port=PORT, debug=False)

# --- build authorize URL ---
def build_authorize_url(*, signup=False, login_hint=None):
    state._verifier, challenge = make_pkce()
    import secrets as _s; state._state = _s.token_urlsafe(16)
    params = {
        "client_id": cfg.client_id, "response_type": "code", "redirect_uri": cfg.redirect_uri,
        "scope": "openid profile email offline_access", "state": state._state,
        "code_challenge": challenge, "code_challenge_method": "S256",
    }
    if AUDIENCE: params["audience"] = AUDIENCE
    if signup: params["screen_hint"] = "signup"
    if login_hint: params["login_hint"] = login_hint
    url = f"{cfg.domain}/authorize?{urlencode(params)}"
    print("DEBUG authorize URL:", url)
    return url

# --- Bridge ---
class Bridge:
    _auth_window = None
    def login(self, email_hint=None):
        url = build_authorize_url(signup=False, login_hint=email_hint)
        _open_auth_modal(url, self); return {"ok": True}
    def signup(self, email_hint=None):
        url = build_authorize_url(signup=True, login_hint=email_hint)
        _open_auth_modal(url, self); return {"ok": True}
    def close_auth(self):
        try:
            if self._auth_window: self._auth_window.destroy(); self._auth_window = None
        except: pass
        return {"ok": True}
    def logout(self):
        state.logged_in=False; state.profile={}; state.access_token=None; state.id_token=None; state.clear_refresh()
        return {"ok": True}
    def profile(self):
        return state.profile if state.logged_in else None
    def try_silent(self):
        if state.load_refresh():
            try: refresh_tokens(cfg, state)
            except: pass
        return {"logged_in": state.logged_in}

def _open_auth_modal(url: str, api: Bridge):
    win = webview.create_window("Sign in — Straight Up", url=url, js_api=api, width=420, height=640, resizable=True)
    api._auth_window = win
    webview.windows.append(win)

# --- main ---
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()

    UI_DIR = APP_DIR.parent / "UI"
    INDEX  = UI_DIR / "index.html"
    print("DEBUG index:", INDEX, "exists?", INDEX.exists(), "uri:", INDEX.as_uri())

    api = Bridge()
    webview.create_window("Straight Up", url=INDEX.as_uri(), js_api=api, width=520, height=700, resizable=True, min_size=(420,560))
    webview.start()
