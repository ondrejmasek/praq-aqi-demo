# 🚀 PrAQ Deployment Guide

The API key has been moved from hardcoded secrets to Streamlit Secrets for secure deployment.

## Local Development

1. **Secrets file already configured:**
   - Location: `.streamlit/secrets.toml`
   - Contains: `API_KEY` and `API_ENDPOINT`
   - Already in `.gitignore` (won't be committed)

2. **Run locally:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Streamlit Cloud Deployment

1. **Go to:** https://share.streamlit.io
2. **Deploy → New app**
   - Repository: `ondrejmasek/praq-aqi-demo`
   - Branch: `main`
   - Main file: `streamlit_app.py`

3. **Configure secrets in Streamlit Cloud:**
   - In your deployed app dashboard → Settings → Secrets
   - Add the following to the secrets editor:
   ```toml
   API_ENDPOINT = "https://praq-maso02-05092205.germanywestcentral.inference.ml.azure.com/score"
   API_KEY = "<your-azure-ml-primary-key-here>"
   ```
   - You can find your API_KEY in Azure ML Studio → Endpoints → your endpoint → Consume tab
   - Click "Save"

4. **Reboot the app** from the settings menu

Your app will then be live at: `https://praq-aqi-demo.streamlit.app/`

---

**Note:** The `.streamlit/secrets.toml` file is local-only and will not be pushed to GitHub (it's in .gitignore).
