"""
app.py
------
The Streamlit web interface for the AI Research Agent.

Run locally with:
    streamlit run app.py
"""

import os
import re

import streamlit as st
from dotenv import load_dotenv

from research_agent import MODEL_NAME, explain_error, run_research

# Load variables from a local .env file (does nothing if the file is missing,
# e.g. on Streamlit Community Cloud).
load_dotenv()


def get_groq_api_key() -> str | None:
    """Find the Groq API key: Streamlit Secrets first, then .env / environment."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        # No secrets.toml file exists -> normal for local development with .env
        pass

    key = os.getenv("GROQ_API_KEY", "").strip()
    return key or None


def make_filename(topic: str) -> str:
    """Turn a topic into a safe file name, e.g. 'research_report_ai_in_education.md'."""
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:60]
    return f"research_report_{slug or 'topic'}.md"


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

st.title("🔎 AI Research Agent")
st.write(
    "Enter a research topic. An AI agent built with **CrewAI** searches the web "
    "(DuckDuckGo, free), analyses what it finds using a **Groq**-hosted LLM, and "
    "writes a structured research report with sources."
)

api_key = get_groq_api_key()
if not api_key:
    st.error(
        "**Groq API key not found.**\n\n"
        "- **Running locally?** Create a file named `.env` in the project folder "
        "containing `GROQ_API_KEY=your_key_here`, then restart the app.\n"
        "- **On Streamlit Community Cloud?** Open your app's **Settings → Secrets** "
        "and add `GROQ_API_KEY = \"your_key_here\"`."
    )
    st.stop()

# Keep the last result between reruns (Streamlit reruns the script on every click).
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.topic = ""

topic = st.text_area(
    "Research topic",
    placeholder="e.g. Impact of Generative AI on Higher Education",
    height=80,
    max_chars=300,
)

if st.button("Research", type="primary", width="stretch"):
    topic = topic.strip()

    if len(topic) < 3:
        st.warning("Please enter a research topic (at least a few words).")
    else:
        st.session_state.result = None
        with st.spinner(
            "The agent is searching the web and writing your report. "
            "This usually takes 1–3 minutes on a free Groq account..."
        ):
            try:
                st.session_state.result = run_research(topic, api_key)
                st.session_state.topic = topic
            except Exception as error:
                st.error(explain_error(error))
                with st.expander("Technical details (for troubleshooting)"):
                    st.exception(error)

# ---------------------------------------------------------------------------
# Show the report
# ---------------------------------------------------------------------------

result = st.session_state.result
if result:
    report = result["report"]
    sources = result["sources"]

    if not report:
        st.error("The agent finished but returned an empty report. Please try again.")
        st.stop()

    st.success("Research complete!")

    if not sources:
        st.warning(
            "The web search did not return any results during this run, so the "
            "report could not be based on retrieved sources. Treat it with caution "
            "and try again in a few minutes."
        )
    elif result["unverified_urls"]:
        st.warning(
            "Some links in the report were **not** returned by the search tool and "
            "may be inaccurate. Please verify them: "
            + ", ".join(result["unverified_urls"])
        )

    st.divider()
    st.markdown(report)
    st.divider()

    with st.expander(f"All {len(sources)} URLs the search tool actually retrieved"):
        if sources:
            for number, source in enumerate(sources, start=1):
                st.markdown(f"{number}. [{source['title']}]({source['url']})")
        else:
            st.write("None.")

    st.download_button(
        label="⬇️ Download Report (Markdown)",
        data=report,
        file_name=make_filename(st.session_state.topic),
        mime="text/markdown",
        width="stretch",
    )

st.caption(f"Model: `{MODEL_NAME}` via Groq · Search: DuckDuckGo (ddgs) · Built with CrewAI + Streamlit")
