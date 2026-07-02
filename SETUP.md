# SHL Assessment Recommender — Setup Guide

An AI-powered API that recommends SHL assessments based on natural-language job descriptions or queries.

---

## Prerequisites

- **Python 3.12+** — [download here](https://www.python.org/downloads/)
- **Groq API key** (free) — [get one here](https://console.groq.com/keys)

---

## Local Setup

1. **Unzip and enter the project folder**

   ```bash
   unzip shl-recommender.zip
   cd project
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   ```

   Open `.env` and replace the placeholder with your Groq API key:

   ```
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

5. **Run the server**

   ```bash
   uvicorn app.main:app --reload
   ```

   On the first run, the embedding model (`all-MiniLM-L6-v2`) and FAISS index are built automatically — this takes ~1-2 minutes. Subsequent starts are instant.

6. **Verify it works**

   - Health check: [http://localhost:8000/health](http://localhost:8000/health)
   - API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Test query:

     ```bash
     curl -X POST http://localhost:8000/recommend \
       -H "Content-Type: application/json" \
       -d '{"query": "I need a test for Java developers"}'
     ```

---

## Deploy to Render (Free Tier)

The project includes a `render.yaml` that handles all configuration.

1. **Push to GitHub**

   ```bash
   git init
   git add .
   git commit -m "initial commit"
   ```

   Create a new repo on GitHub, then:

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/shl-recommender.git
   git push -u origin main
   ```

2. **Connect to Render**

   - Go to [render.com](https://render.com) and sign up / log in.
   - Click **New > Web Service** and connect your GitHub repo.
   - Render will auto-detect `render.yaml` and configure the service.

3. **Set the environment variable**

   - In the Render dashboard, go to **Environment** and add:
     - `GROQ_API_KEY` = your Groq API key

4. **Deploy**

   Render will install dependencies and start the service. First deploy takes a few minutes (model download).

5. **Verify**

   ```
   https://shl-recommender.onrender.com/health
   https://shl-recommender.onrender.com/docs
   ```

---

## Notes

- **Cold starts**: Render's free tier spins down after ~15 min of inactivity. The first request after that takes ~30-60 seconds.
- **No Docker needed**: Render uses the Python runtime directly via `render.yaml`.
- **Model download**: The sentence-transformers model (~90 MB) downloads on first start. It's cached after that.
- **FAISS index**: Built in-memory from the assessment catalog on startup — no external database needed.
