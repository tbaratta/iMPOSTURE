# minimal helpers for token exchange (PKCE + rotation)
import os, secrets, base64, hashlib, keyring
from dataclasses import dataclass
from authlib.integrations.requests_client import OAuth2Session

SERVICE_NAME = "StraightUp"

def _b64url(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

def make_pkce():
    verifier = _b64url(os.urandom(40))
    import hashlib
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge

@dataclass
class Auth0Config:
    domain: str
    client_id: str
    redirect_uri: str
    audience: str | None = None

def _auth0_base_from(cfg: "Auth0Config") -> str:
    d = (cfg.domain or "").strip().rstrip("/")
    return d if d.startswith("http") else f"https://{d}"

class AuthState:
    def __init__(self):
        self.logged_in = False
        self.profile = {}
        self.access_token = None
        self.id_token = None
        self._refresh_token = None
        self._verifier = None
        self._state = None
        self._session: OAuth2Session | None = None
    def load_refresh(self, username_hint="default"):
        t = keyring.get_password(SERVICE_NAME, username_hint)
        if t: self._refresh_token = t
        return t
    def save_refresh(self, token, username_hint="default"):
        keyring.set_password(SERVICE_NAME, username_hint, token)
    def clear_refresh(self, username_hint="default"):
        try: keyring.delete_password(SERVICE_NAME, username_hint)
        except Exception: pass

def exchange_code_for_tokens(cfg: Auth0Config, st: AuthState, code: str, state: str):
    if state != st._state: raise RuntimeError("State mismatch")
    base = _auth0_base_from(cfg)
    token_url = f"{base}/oauth/token"
    st._session = OAuth2Session(client_id=cfg.client_id, redirect_uri=cfg.redirect_uri)
    token = st._session.fetch_token(
        url=token_url, grant_type="authorization_code",
        code=code, code_verifier=st._verifier)
    st.access_token = token.get("access_token")
    st.id_token     = token.get("id_token")
    rt = token.get("refresh_token")
    if rt: st._refresh_token = rt; st.save_refresh(rt)
    ui = st._session.get(f"{base}/userinfo",
        headers={"Authorization": f"Bearer {st.access_token}"}).json()
    st.profile = ui; st.logged_in = True

def refresh_tokens(cfg: Auth0Config, st: AuthState):
    if not st._refresh_token: raise RuntimeError("No refresh token")
    base = _auth0_base_from(cfg); token_url = f"{base}/oauth/token"
    st._session = OAuth2Session(client_id=cfg.client_id, redirect_uri=cfg.redirect_uri)
    token = st._session.fetch_token(
        url=token_url, grant_type="refresh_token", refresh_token=st._refresh_token)
    st.access_token = token.get("access_token")
    st.id_token     = token.get("id_token") or st.id_token
    new_rt = token.get("refresh_token")
    if new_rt and new_rt != st._refresh_token:
        st._refresh_token = new_rt; st.save_refresh(new_rt)

def build_logout_url(cfg: Auth0Config):
    from urllib.parse import urlencode
    base = _auth0_base_from(cfg)
    params = urlencode({
        "client_id": cfg.client_id,
        "returnTo": cfg.redirect_uri.replace("/callback", "/logout-complete")
    })
    return f"{base}/v2/logout?{params}"
