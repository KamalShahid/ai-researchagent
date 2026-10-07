"""
research_agent.py
-----------------
Everything related to the AI side of the project lives here:

  1. A free web-search TOOL (DuckDuckGo via the `ddgs` package)
  2. The Groq LLM connection
  3. One CrewAI AGENT (the "Research Agent")
  4. One CrewAI TASK (write the research report)
  5. A CREW that runs the agent on the task

The Streamlit app (app.py) only calls `run_research(topic, api_key)`.
"""

import os

# --- Turn off CrewAI's optional tracing/telemetry --------------------------
# These must be set BEFORE crewai is imported. They stop CrewAI from asking
# "Would you like to view your execution traces?" in the terminal (which can
# pause the app for ~20 seconds) and from sending anonymous usage data.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import re
from datetime import date
from typing import Type

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Settings you can change
# ---------------------------------------------------------------------------

# "groq/" tells CrewAI (through LiteLLM) to use Groq.
# "openai/gpt-oss-120b" is the exact model ID listed in the Groq console.
MODEL_NAME = "groq/openai/gpt-oss-120b"

# How many results each web search returns, and how much of each snippet
# we keep. Small numbers keep each request under Groq's free-tier
# tokens-per-minute limit.
RESULTS_PER_SEARCH = 5
SNIPPET_MAX_CHARS = 300

# Maximum LLM calls per minute made by the agent. CrewAI waits automatically
# when this is reached. This keeps a free Groq account under its
# tokens-per-minute limit. If you upgrade your Groq plan, set it to None.
MAX_REQUESTS_PER_MINUTE = 3


# ---------------------------------------------------------------------------
# 1. The web-search TOOL
# ---------------------------------------------------------------------------

class WebSearchInput(BaseModel):
    """What the agent must provide when it calls the tool."""

    query: str = Field(
        ...,
        description="A short, specific web search query, e.g. "
        "'generative AI impact on university assessment 2026'.",
    )


class WebSearchTool(BaseTool):
    """Searches the web for free (no API key) using the ddgs package."""

    name: str = "web_search"
    description: str = (
        "Searches the web and returns up to 5 results, each with a title, "
        "URL and short snippet. Use it to find current, factual information "
        "about the research topic. Input: a short search query."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    # Every URL the tool actually returned during this run.
    # The app shows this list so you can check the report's references.
    sources: list[dict] = Field(default_factory=list)

    def _run(self, query: str) -> str:
        try:
            results = DDGS().text(query, max_results=RESULTS_PER_SEARCH)
        except RatelimitException:
            return (
                "SEARCH ERROR: the search engine is rate-limiting requests. "
                "Do not retry more than once; write the report with the "
                "information you already have."
            )
        except TimeoutException:
            return "SEARCH ERROR: the search timed out. Try one different, shorter query."
        except DDGSException as error:
            return f"SEARCH ERROR: no results could be retrieved ({error})."
        except Exception as error:  # e.g. no internet connection
            return f"SEARCH ERROR: unexpected problem while searching ({error})."

        if not results:
            return f"No results found for '{query}'. Try a different query."

        known_urls = {source["url"] for source in self.sources}
        lines = []
        for number, result in enumerate(results, start=1):
            title = (result.get("title") or "Untitled").strip()
            url = (result.get("href") or "").strip()
            snippet = (result.get("body") or "").strip()[:SNIPPET_MAX_CHARS]

            if url and url not in known_urls:
                self.sources.append({"title": title, "url": url})
                known_urls.add(url)

            lines.append(f"[{number}] {title}\nURL: {url}\nSnippet: {snippet}")

        return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 2-5. LLM, Agent, Task and Crew
# ---------------------------------------------------------------------------

def build_crew(api_key: str, search_tool: WebSearchTool) -> Crew:
    """Create the LLM, the Research Agent, its Task, and the Crew."""

    # 2. The LLM (Groq)
    llm = LLM(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.3,         # lower = more factual, less creative
        max_tokens=2500,         # enough for a ~800-1,200 word report
        reasoning_effort="low",  # gpt-oss "thinks" less -> faster, fewer tokens
    )

    # 3. The Agent
    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            "Research the topic '{topic}' on the web and turn the findings "
            "into a clear, accurate, well-structured research report."
        ),
        backstory=(
            "You are an experienced research analyst. You search for "
            "reliable sources (universities, research institutes, government "
            "bodies, international organisations and reputable news outlets), "
            "compare what they say, and explain it in your own words. You "
            "never invent facts, statistics or references."
        ),
        tools=[search_tool],
        llm=llm,
        allow_delegation=False,  # only one agent, nobody to delegate to
        max_iter=8,              # safety limit on think/search cycles
        max_rpm=MAX_REQUESTS_PER_MINUTE,
        verbose=True,            # prints the agent's steps in the terminal
    )

    # 4. The Task
    research_task = Task(
        description=(
            "Research this topic: '{topic}'. Today's date is {today}.\n\n"
            "Steps:\n"
            "1. Use the web_search tool 2 to 4 times with DIFFERENT queries "
            "that cover different angles of the topic (overview, recent "
            "developments, benefits, risks, statistics).\n"
            "2. Prefer credible sources and recent information.\n"
            "3. Analyse and synthesise what you found. Write in your own "
            "words; do not copy snippets.\n"
            "4. Write the final report in Markdown.\n\n"
            "Rules about sources (very important):\n"
            "- Only cite URLs that appeared in web_search results during "
            "this task. Never invent a URL, title, author or statistic.\n"
            "- Refer to sources in the text with numbers like [1], [2] that "
            "match the Sources / References list.\n"
            "- If the searches failed or returned nothing useful, say so "
            "clearly in the report, write the analysis from general "
            "knowledge marked as such, and put 'No sources could be "
            "retrieved for this report.' under Sources / References."
        ),
        expected_output=(
            "A Markdown research report of roughly 800-1,200 words using "
            "this structure (headings may be adapted slightly to the topic):\n\n"
            "# Research Report: <topic>\n"
            "## Executive Summary\n"
            "## Introduction\n"
            "## Key Findings  (bullet points)\n"
            "## Detailed Analysis\n"
            "## Challenges / Limitations\n"
            "## Future Outlook\n"
            "## Conclusion\n"
            "## Sources / References  (numbered list: [n] Title - URL)\n\n"
            "Return only the report, with no text before or after it."
        ),
        agent=researcher,
    )

    # 5. The Crew
    return Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        tracing=False,
        verbose=False,
    )


def run_research(topic: str, api_key: str) -> dict:
    """
    Run the Research Agent on a topic.

    Returns a dictionary with:
      report           - the Markdown report written by the agent
      sources          - every URL the search tool really returned
      unverified_urls  - URLs in the report that the search tool never returned
    """
    search_tool = WebSearchTool()  # a fresh tool (and source list) per run
    crew = build_crew(api_key, search_tool)

    result = crew.kickoff(
        inputs={"topic": topic, "today": date.today().strftime("%d %B %Y")}
    )
    report = (result.raw or "").strip()

    return {
        "report": report,
        "sources": search_tool.sources,
        "unverified_urls": find_unverified_urls(report, search_tool.sources),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_unverified_urls(report: str, sources: list[dict]) -> list[str]:
    """List URLs in the report that were never returned by the search tool."""
    found_in_report = re.findall(r"https?://[^\s\)\]>\"'<]+", report)
    retrieved = {source["url"].rstrip("/.,") for source in sources}

    unverified = []
    for url in found_in_report:
        clean_url = url.rstrip("/.,;:")
        if clean_url not in retrieved and clean_url not in unverified:
            unverified.append(clean_url)
    return unverified


def explain_error(error: Exception) -> str:
    """Turn a technical error into a beginner-friendly message."""
    text = f"{type(error).__name__}: {error}".lower()

    if "authentication" in text or "invalid api key" in text or "invalid_api_key" in text or "401" in text:
        return (
            "Groq rejected the API key (invalid or revoked). Create a new key "
            "at https://console.groq.com/keys and update your .env file or "
            "Streamlit secrets."
        )
    if "request too large" in text or "413" in text:
        return (
            "The request was too large for your Groq plan's tokens-per-minute "
            "limit. Try a narrower topic, or lower RESULTS_PER_SEARCH in "
            "research_agent.py."
        )
    if "ratelimit" in text or "rate limit" in text or "429" in text:
        return (
            "Groq's rate limit was reached (the free tier allows a limited "
            "number of tokens per minute and per day). Wait about one minute "
            "and try again. If it keeps happening, you may have used your "
            "daily allowance."
        )
    if "model_not_found" in text or "does not exist" in text or "decommissioned" in text or "404" in text:
        return (
            f"The model '{MODEL_NAME}' was not found on Groq. Check the "
            "current model list at https://console.groq.com/docs/models "
            "and update MODEL_NAME in research_agent.py."
        )
    if "litellm" in text and ("not installed" in text or "fallback" in text):
        return (
            "LiteLLM is missing. Run: pip install -r requirements.txt "
            "(it installs crewai[litellm], which CrewAI needs for Groq)."
        )
    if "connection" in text or "timeout" in text or "timed out" in text:
        return (
            "Could not reach Groq (network problem or timeout). Check your "
            "internet connection and try again."
        )
    return f"Something went wrong while running the agent: {type(error).__name__}: {error}"
