# StraightUp Desktop - Supabase Authentication Setup

## Overview
Your tkinter desktop app now includes Supabase authentication with login/signup screens, session management, and user authentication gating.

## Features Added
- ✅ **Login Screen**: Email/password authentication
- ✅ **Signup Screen**: Create new accounts with email verification
- ✅ **Session Management**: Persistent authentication state
- ✅ **Logout Function**: Clean session termination
- ✅ **Auth Gate**: App only accessible after login
- ✅ **User Display**: Shows logged-in user email in header

## Setup Instructions

### 1. Install Dependencies
Dependencies are already installed:
```bash
pip install supabase python-dotenv
```

### 2. Create Supabase Project
1. Go to [https://supabase.com/dashboard](https://supabase.com/dashboard)
2. Create a new project or select existing one
3. Wait for project setup to complete

### 3. Get API Credentials
1. In your Supabase dashboard, go to **Settings > API**
2. Copy the following values:
   - **Project URL** (looks like: `https://xxx.supabase.co`)
   - **anon public key** (long string starting with `eyJ...`)

### 4. Create Environment File
1. Copy the example file:
   ```bash
   copy .env.example .env
   ```
2. Edit `.env` file with your actual Supabase credentials:
   ```
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

### 5. Enable Email Authentication
In your Supabase dashboard:
1. Go to **Authentication > Settings**
2. Ensure **Enable email confirmations** is configured as needed
3. For development, you can disable email confirmation temporarily

## How It Works

### App Flow
1. **App Start**: Checks if user is authenticated
2. **Not Authenticated**: Shows login/signup screens
3. **Authenticated**: Shows main app with user info and logout button

### Authentication Screens
- **Login**: Email/password with signup link
- **Signup**: Create account with login link
- **Error Handling**: Shows authentication errors clearly

### Main App Changes
- **Header**: Now includes user email and logout button
- **Session**: Maintains authentication state throughout app usage
- **Logout**: Cleans session and returns to login screen

## Testing Without Supabase
If you don't have Supabase set up yet:
- App will show "Authentication not available" message
- You can still test the UI screens
- All auth functions will gracefully handle missing setup

## Files Modified
- `tkinter_app.py`: Added full authentication system
- `.env.example`: Template for credentials
- This README for setup instructions

## Security Notes
- ✅ Environment variables keep credentials secure
- ✅ Session state properly managed
- ✅ Logout cleans authentication state
- ✅ Auth gate prevents unauthorized access