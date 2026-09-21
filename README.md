# H2H Web Chat — Vercel + Kaggle Ollama

Vercel hosts the web UI/Flask API. Gemma 4:26B stays on Kaggle GPU.
Ollama Web Search runs through ollama.com.

## Vercel environment variables

Set these in Vercel Project Settings → Environment Variables:

OLLAMA_HOST=https://YOUR-CLOUDFLARE-QUICK-TUNNEL.trycloudflare.com
OLLAMA_API_KEY=ollama_xxxxx

Important:
- Never put the API key in frontend JavaScript.
- The Cloudflare Quick Tunnel URL is temporary.
- When Kaggle restarts and a new tunnel URL is created, update OLLAMA_HOST
  in Vercel and redeploy/re-run the deployment.
- The Flask app only binds to Vercel's server runtime in production.

## Deploy with Vercel CLI

npm i -g vercel
vercel login
cd h2h-web-vercel
vercel

For production:
vercel --prod

## GitHub deployment

Push this folder to GitHub, then import the repository in Vercel.
Add the two environment variables above before deploying.

## Local test

python -m pip install -r requirements.txt

Windows CMD:
set OLLAMA_HOST=https://YOUR-QUICK-TUNNEL.trycloudflare.com
set OLLAMA_API_KEY=YOUR_KEY

python app.py

Open http://127.0.0.1:5000
