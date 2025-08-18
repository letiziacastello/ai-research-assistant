import os
from datetime import datetime
from typing import List

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

# LangChain Core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory

# Custom tools
from tools import (
    search_tool,
    wiki_tool,
    save_tool,
    translate_tool,
    summarize_tool,
    export_to_pdf_tool,
    scrape_tool,
)

# =========================
# Environment & UI Settings
# =========================
load_dotenv()
st.set_page_config(page_title="AI Research Assistant", layout="wide")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_history" not in st.session_state:
    st.session_state.selected_history = None


# =========================
# Helper Functions
# =========================
def tool_chip(name: str) -> str:
    """Return an HTML badge for the tool name with icon, using external CSS."""
    n = name.lower()
    icon = (
        "🔎" if "search" in n else
        "📚" if "wiki" in n else
        "📝" if "summarize" in n else
        "🌐" if "translate" in n else
        "📄" if "pdf" in n else
        "💾" if "save" in n else
        "🧩"
    )
    return f"<span class='tool-badge'>{icon} {name}</span>"

def linkify_sources(sources):
    """Convert list of sources into clickable HTML links."""
    html_items = []
    for s in sources:
        s = s.strip()
        url = s if s.startswith("http") else None
        label = s if not url else (s.split('//', 1)[-1][:60] + ('…' if len(s) > 60 else ''))
        html_items.append(
            f"<li><a href='{url or '#'}' target='_blank' rel='noopener noreferrer'>{label}</a></li>"
            if url else f"<li>{label}</li>"
        )
    return "<ul>" + "".join(html_items) + "</ul>" if html_items else "—"


def show_loader():
    """Display a custom animated loader in the UI."""
    placeholder = st.empty()
    loader_html = """
    <div style="display:flex;align-items:center;gap:14px;margin:20px 0;">
      <div class="loader-dots"></div>
      <div style="font-weight:600;font-size:16px;">The AI is gathering knowledge for you...</div>
    </div>
    <style>
    .loader-dots {
      width: 64px; height: 14px;
      background:
        radial-gradient(circle 6px, #ff6ec4 95%, transparent) 0 50%,
        radial-gradient(circle 6px, #7873f5 95%, transparent) 50% 50%,
        radial-gradient(circle 6px, #4ade80 95%, transparent) 100% 50%;
      background-size: 14px 14px;
      background-repeat: no-repeat;
      animation: jump 0.9s linear infinite;
      filter: drop-shadow(0 0 6px rgba(255,110,196,.6));
    }
    @keyframes jump {
      0%   { background-position:   0 50%, 50% 50%, 100% 50%; }
      33%  { background-position:   0 0%,  50% 50%, 100% 50%; }
      66%  { background-position:   0 50%, 50% 0%,  100% 50%; }
      100% { background-position:   0 50%, 50% 50%, 100% 0%;  }
    }
    </style>
    """
    placeholder.markdown(loader_html, unsafe_allow_html=True)
    return placeholder


def load_custom_css(file_path: str):
    """Load and apply custom CSS if file exists."""
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =========================
# Sidebar: Theme & Options
# =========================
st.sidebar.header("🎨 Theme")
theme_choice = st.sidebar.radio("Select Theme:", ["Light", "Dark"], index=1)
if theme_choice == "Dark":
    load_custom_css("style.css")

st.sidebar.header("⚙️ Options")
output_language = st.sidebar.selectbox(
    "Output language (for translation):",
    ["Italian", "English", "French", "Spanish"],
    index=0
)
use_wikipedia = st.sidebar.checkbox("Use Wikipedia", value=True)
use_search = st.sidebar.checkbox("Use Web Search (DuckDuckGo)", value=True)
do_summarize = st.sidebar.checkbox("Summarize", value=True)
do_translate = st.sidebar.checkbox("Translate", value=True)
do_pdf = st.sidebar.checkbox("Export PDF", value=True)
show_original = st.sidebar.checkbox("Show original text", value=True)
with st.sidebar.expander("Optional: scrape a URL"):
    scrape_url = st.text_input("URL to scrape (https://...)", value="")
st.sidebar.caption("Tip: install `ddgs` to silence DuckDuckGo warning.")

# =========================
# Sidebar: History
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("📜 Search history")

def _hist_label(item):
    """Format history items for display in the sidebar."""
    when = item.get("timestamp", "")
    topic = item.get("topic", "—")
    lang = item.get("language", "")
    return f"{topic} · {when} · {lang}"

# Show recent history
hist_items = list(reversed(st.session_state.history))[:10]
choices = [_hist_label(it) for it in hist_items]
selected = st.sidebar.selectbox("Previous runs:", ["—"] + choices, index=0)

# History control buttons
colh1, colh2 = st.sidebar.columns(2)
if colh1.button("Load") and selected != "—":
    st.session_state.selected_history = hist_items[choices.index(selected)]
if colh2.button("Clear"):
    st.session_state.history.clear()
    st.session_state.selected_history = None

# Cost-saver mode
cost_saver = st.sidebar.toggle("💸 Cost saver (cheaper model & fewer tools)", value=True)
model_name = "gpt-4o-mini" if cost_saver else "gpt-4o"
llm = ChatOpenAI(model=model_name, temperature=0)

# =========================
# Prompt & Output Model
# =========================
class ResearchOutput(BaseModel):
    topic: str
    summary: str
    sources: List[str]
    tools_used: List[str]

parser = PydanticOutputParser(pydantic_object=ResearchOutput)
format_instructions = parser.get_format_instructions()

system_message = """
# ROLE:
You are a highly capable AI Research Assistant.
You generate structured research outputs and call tools when needed.

# TASK:
- Research the given topic, synthesize a fluent, well-written paragraph (no bullet points), and produce a structured JSON.

# OUTPUT FORMAT:
{format_instructions}

# CONSTRAINTS:
- Output ONLY valid JSON (no markdown, no extra text).
- Use tools only if relevant.
- Do not invent sources; list only sources truly used.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=format_instructions)

# =========================
# User Input Form
# =========================
st.title("AI Research Assistant")

with st.form("research_form"):
    query = st.text_input("What do you want to research?", key="query")
    run = st.form_submit_button("Run Research")

# =========================
# Agent Builder
# =========================
def build_agent(selected_tools):
    """Create a tool-calling agent with memory."""
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=selected_tools)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return AgentExecutor(
        agent=agent,
        tools=selected_tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        return_only_outputs=True,
    )

# =========================
# Execution
# =========================
if run and query:
    # Build the intent string for the agent
    user_intent = query.strip()
    if do_summarize and not cost_saver:
        user_intent += "\n\nPlease summarize the findings in one fluent paragraph."
    if scrape_url:
        user_intent += f"\n\nAlso, scrape and use content from this URL if relevant: {scrape_url}"

    # Select tools dynamically based on settings
    selected_tools = []
    if use_search and not cost_saver:
        selected_tools.append(search_tool)
    if use_wikipedia:
        selected_tools.append(wiki_tool)
    if do_summarize and not cost_saver:
        selected_tools.append(summarize_tool)
    selected_tools.append(save_tool)
    if scrape_url and not cost_saver:
        selected_tools.append(scrape_tool)
    if do_pdf:
        selected_tools.append(export_to_pdf_tool)

    executor = build_agent(selected_tools)

    loader = None
    raw_response = {}
    try:
        # Show loader
        loader = show_loader()

        # Run the agent
        raw_response = executor.invoke({"input": user_intent})
        st.session_state["last_steps"] = raw_response.get("intermediate_steps", [])

        # Hide loader
        if loader:
            loader.empty()

        # Parse the structured output
        data = parser.parse(raw_response["output"])
        original_summary = data.summary

        # Post-processing: translation
        if do_translate and output_language.lower() != "english":
            try:
                translated = translate_tool.func(data.summary, output_language)
                if translated.strip():
                    data.summary = translated.strip()
                    data.tools_used.append(f"post_translate({output_language})")
            except Exception as e:
                st.warning(f"Translation step failed: {e}")

        # ===== Display Output =====
        lang_badge = (
            f"<span style='font-size:12px;padding:4px 8px;border:1px solid #444;border-radius:999px;margin-left:8px;'>Translated to {output_language}</span>"
            if do_translate and output_language.lower() != "english"
            else ""
        )
        st.markdown(f"<h2 style='margin-top:0'>📌 {data.topic} {lang_badge}</h2>", unsafe_allow_html=True)

        st.markdown("<h3>📝 Summary</h3>", unsafe_allow_html=True)
        if do_translate and output_language.lower() != "english" and show_original:
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Original")
                st.markdown(original_summary, unsafe_allow_html=True)
            with col2:
                st.caption(f"Translated to {output_language}")
                st.markdown(data.summary, unsafe_allow_html=True)
        else:
            st.markdown(data.summary, unsafe_allow_html=True)

        st.markdown("<h3>🛠 Tools Used</h3>", unsafe_allow_html=True)
        st.markdown(" ".join(tool_chip(t) for t in data.tools_used) or "—", unsafe_allow_html=True)

        st.markdown("<h3>📚 Sources</h3>", unsafe_allow_html=True)
        st.markdown(linkify_sources(data.sources), unsafe_allow_html=True)

        # ===== Save Output =====
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_topic = data.topic.replace(" ", "_")
        txt_path = os.path.join("outputs", f"research_{safe_topic}_{timestamp}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Topic: {data.topic}\n\nSummary:\n{data.summary}\n\nSources:\n" + "\n".join(data.sources))

        st.download_button("⬇️ Download TXT", data=open(txt_path, "rb"), file_name=os.path.basename(txt_path))

        if do_pdf:
            pdf_path = os.path.join("outputs", f"report_{safe_topic}_{timestamp}.pdf")
            export_to_pdf_tool.run({"text": data.summary, "filename": pdf_path})
            st.download_button("⬇️ Download PDF", data=open(pdf_path, "rb"), file_name=os.path.basename(pdf_path))

        # Save run in history
        st.session_state.history.append({
            "topic": data.topic,
            "summary": data.summary,
            "original_summary": original_summary,
            "language": output_language if do_translate else "English",
            "sources": data.sources,
            "tools_used": data.tools_used,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "txt_path": txt_path,
            "pdf_path": pdf_path if do_pdf else None,
        })

    except Exception as e:
        if loader:
            loader.empty()
        st.error(f"Parsing or agent error: {e}")
        st.code(raw_response.get("output", "No output"))

# =========================
# Debug Info
# =========================
with st.expander("🔍 Debug / Technical details"):
    steps = st.session_state.get("last_steps", [])
    if steps:
        for i, (action, observation) in enumerate(steps, start=1):
            st.markdown(f"**Step {i} – Tool:** `{getattr(action, 'tool', 'unknown')}`")
            st.code(str(getattr(action, 'tool_input', ''))[:1000])
            st.write(observation if isinstance(observation, str) else str(observation))
    else:
        st.write("No intermediate steps captured (maybe no tools were used).")