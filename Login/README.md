# Straight Up — Login Setup

## Steps to Run

1. Open a terminal and go to the Login/Auth0 folder:
   cd StraightUp/Login/Auth0

2. Create and activate a virtual environment:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   (If activation is blocked, run this once:)
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force

3. Install dependencies:
   pip install -r requirements.txt

4. Copy the example env file:
   copy .env.example .env

   Then edit .env and paste in the Auth0 values we’ll share with you.

5. Run the app:
   python app.py

## What Happens
- A desktop window will open with the Straight Up login page.
- Click **Sign In** or **Create Account** to go through Auth0.
- Accounts and credentials are stored in Auth0, you don’t need a separate database.

That’s all you need to get it working locally.
