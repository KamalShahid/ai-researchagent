# 🔎 AI Research Agent

A beginner-friendly, single-agent research assistant.

**Topic → web search → analysis → structured Markdown report → shown in Streamlit**

| Part | Technology |
|---|---|
| Agent framework | CrewAI 1.15 (Agent, Task, Crew, Tool) |
| LLM | Groq — `openai/gpt-oss-120b` (via LiteLLM) |
| Web search | `ddgs` (DuckDuckGo metasearch, free, no API key) |
| Frontend | Streamlit |
| Hosting | GitHub + Streamlit Community Cloud |

## Project structure

```text
ai-research-agent/
├── app.py                         # Streamlit user interface
├── research_agent.py              # Search tool + Groq LLM + Agent + Task + Crew
├── requirements.txt               # Pinned, tested dependency versions
├── .env.example                   # Template for your local .env file
├── .gitignore                     # Keeps secrets and venv out of GitHub
├── .streamlit/
│   └── secrets.toml.example       # Template for Streamlit Cloud secrets
└── README.md
```

## Run locally (macOS)

Requires **Python 3.12** (CrewAI does not support Python 3.14 yet).

```bash
cd ai-research-agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env          # then edit .env and paste your Groq key
streamlit run app.py
```

Get a free Groq API key at <https://console.groq.com/keys>.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (the `.env` file is ignored automatically).
2. Go to <https://share.streamlit.io> → **Create app** → choose the repo, branch `main`, main file `app.py`.
3. Under **Advanced settings**, choose **Python 3.12** and paste into **Secrets**:

   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

4. Click **Deploy**.

## Notes

- On Groq's free tier the agent is paced to stay under the tokens-per-minute limit, so a report usually takes 1–3 minutes.
- The app lists every URL the search tool really returned and warns if the report cites a link that was never retrieved.
- Change the model, number of search results or pacing at the top of `research_agent.py`.
