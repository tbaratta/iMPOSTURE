// app.js — works for index.html (sign in) and register.html (create account)

const emailInput  = document.getElementById('email');
const passInput   = document.getElementById('password');
const nameInput   = document.getElementById('nameIn');

const btnLogin    = document.getElementById('btnLogin');
const btnSignup   = document.getElementById('btnSignup');

const gotoSignin  = document.getElementById('gotoSignin');

const statusText  = document.getElementById('statusText');
const profileDiv  = document.getElementById('profile');
const nameSpan    = document.getElementById('name');
const emailOut    = document.getElementById('emailOut');

async function refreshStatus() {
  try {
    const res = await fetch('http://127.0.0.1:8765/status');
    const data = await res.json();
    if (statusText) {
      if (data.logged_in) {
        statusText.textContent = 'Signed in';
        if (profileDiv) profileDiv.classList.remove('hidden');
        if (nameSpan)   nameSpan.textContent  = data.profile?.name  ?? '—';
        if (emailOut)   emailOut.textContent  = data.profile?.email ?? '—';
      } else {
        statusText.textContent = 'Signed out';
        if (profileDiv) profileDiv.classList.add('hidden');
        if (nameSpan)   nameSpan.textContent = '—';
        if (emailOut)   emailOut.textContent = '—';
      }
    }
  } catch {}
}

// Sign in (opens embedded Universal Login - login tab)
if (btnLogin) {
  btnLogin.addEventListener('click', async () => {
    if (statusText) statusText.textContent = 'Opening…';
    const hint = emailInput?.value || null;
    await window.pywebview.api.login(hint);
    const t0 = Date.now();
    const iv = setInterval(async () => {
      await refreshStatus();
      if ((statusText && statusText.textContent === 'Signed in') || Date.now() - t0 > 15000) {
        clearInterval(iv);
      }
    }, 800);
  });
}

// Create account (opens embedded Universal Login - signup tab)
if (btnSignup) {
  btnSignup.addEventListener('click', async () => {
    if (statusText) statusText.textContent = 'Opening…';
    const hint = emailInput?.value || null;
    await window.pywebview.api.signup(hint);
    const t0 = Date.now();
    const iv = setInterval(async () => {
      await refreshStatus();
      if ((statusText && statusText.textContent === 'Signed in') || Date.now() - t0 > 20000) {
        clearInterval(iv);
      }
    }, 800);
  });
}

// Back to sign in (from register.html)
if (gotoSignin) {
  gotoSignin.addEventListener('click', () => {
    window.location.href = 'index.html';
  });
}

window.addEventListener('DOMContentLoaded', async () => {
  if (window.pywebview?.api?.try_silent) {
    await window.pywebview.api.try_silent();
  }
  await refreshStatus();
});
