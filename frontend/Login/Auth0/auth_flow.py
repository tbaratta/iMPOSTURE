# auth_flow.py — PKCE login + refresh + secure refresh token storage

import os, secrets, base64, hashlib, webbrowser, keyring
from dataclasses import dataclass
from urllib.parse import urlencode
from authlib.integrations.requests_client import OAuth2Session

SERVICE_NAME = "StraightUp"

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

def make_pkce():
    verifier = _b64url(os.urandom(40))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge

@dataclass
class Auth0Config:
    domain: str
    client_id: str
    redirect_uri: str
    audience: str | None = None

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
        token = keyring.get_password(SERVICE_NAME, username_hint)
        if token:
            self._refresh_token = token
        return token

    def save_refresh(self, token, username_hint="default"):
        keyring.set_password(SERVICE_NAME, username_hint, token)

    def clear_refresh(self, username_hint="default"):
        try:
            keyring.delete_password(SERVICE_NAME, username_hint)
        except Exception:
            pass

def start_login(cfg: Auth0Config, st: AuthState):
    st._verifier, challenge = make_pkce()
    st._state = secrets.token_urlsafe(16)
    params = {
        "client_id": cfg.client_id,
        "response_type": "code",
        "redirect_uri": cfg.redirect_uri,
        "scope": "openid profile email offline_access",
        "state": st._state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if cfg.audience:
        params["audience"] = cfg.audience
    url = f"{cfg.domain}/authorize?{urlencode(params)}"
    webbrowser.open(url)

def exchange_code_for_tokens(cfg: Auth0Config, st: AuthState, code: str, state: str):
    if state != st._state:
        raise RuntimeError("State mismatch")
    token_url = f"{cfg.domain}/oauth/token"
    st._session = OAuth2Session(client_id=cfg.client_id, redirect_uri=cfg.redirect_uri)
    token = st._session.fetch_token(
        url=token_url,
        grant_type="authorization_code",
        code=code,
        code_verifier=st._verifier,
    )
    st.access_token = token.get("access_token")
    st.id_token     = token.get("id_token")

    rt = token.get("refresh_token")
    if rt:
        st._refresh_token = rt
        st.save_refresh(rt)

    ui = st._session.get(
        f"{cfg.domain}/userinfo",
        headers={"Authorization": f"Bearer {st.access_token}"}
    ).json()

    st.profile = ui  # includes sub, name, email (name may be empty)
    st.logged_in = True

def refresh_tokens(cfg: Auth0Config, st: AuthState):
    if not st._refresh_token:
        raise RuntimeError("No refresh token")
    token_url = f"{cfg.domain}/oauth/token"
    st._session = OAuth2Session(client_id=cfg.client_id, redirect_uri=cfg.redirect_uri)
    token = st._session.fetch_token(
        url=token_url,
        grant_type="refresh_token",
        refresh_token=st._refresh_token,
    )
    st.access_token = token.get("access_token")
    st.id_token     = token.get("id_token") or st.id_token
    new_rt = token.get("refresh_token")
    if new_rt and new_rt != st._refresh_token:
        st._refresh_token = new_rt
        st.save_refresh(new_rt)
