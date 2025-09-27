// Minimal login page logic: open Auth0 modal, then poll /status and go to home.html

async function checkAndGoHome() {
  try {
    const r = await fetch('http://127.0.0.1:8765/status', { cache: 'no-store' });
    const j = await r.json();
    if (j.logged_in) {
      window.location.href = 'home.html';
      return true;
    }
  } catch (e) {
    // ignore transient errors
  }
  return false;
}

document.getElementById('btnLogin').onclick = async () => {
  await window.pywebview.api.login(null);
  // Poll for up to ~20s; stop when logged in
  const t0 = Date.now();
  const iv = setInterval(async () => {
    if (await checkAndGoHome() || Date.now() - t0 > 20000) clearInterval(iv);
  }, 700);
};

document.getElementById('btnSignup').onclick = async () => {
  await window.pywebview.api.signup(null);
  const t0 = Date.now();
  const iv = setInterval(async () => {
    if (await checkAndGoHome() || Date.now() - t0 > 25000) clearInterval(iv);
  }, 700);
};

// If refresh token exists, silently sign in and jump straight to home
window.addEventListener('DOMContentLoaded', async () => {
  if (window.pywebview?.api?.try_silent) {
    try { await window.pywebview.api.try_silent(); } catch {}
  }
  await checkAndGoHome();
});
