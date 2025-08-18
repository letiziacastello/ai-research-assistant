from datetime import datetime
from typing import Optional
import json
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI

load_dotenv()

# -----------------------------
# LLM for lightweight tasks (translations, summaries, etc.)
# -----------------------------
llm_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -----------------------------
# Save results to TXT
# -----------------------------
def save_to_txt(data: dict, filename: str = "research_output.txt") -> str:
    """Append a structured research result as JSON to a text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_text = json.dumps(data, indent=2, ensure_ascii=False)
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{json_text}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Append the complete structured research result (topic, summary, sources, tools_used) to a local text file."
)

# -----------------------------
# Web search (DuckDuckGo)
# -----------------------------
search_tool = Tool(
    name="search",
    func=DuckDuckGoSearchRun().run,
    description="Search the web for recent or general information. Input is a search query."
)

# -----------------------------
# Wikipedia
# -----------------------------
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500)
)

# -----------------------------
# Translate
# -----------------------------
def translate_text(text: str, target_language: Optional[str] = "Italian") -> str:
    """Translate text to the specified target language."""
    prompt = (
        f"Translate the following text into {target_language}. "
        "Return ONLY the translation, with no preface or quotes.\n\n"
        f"Text:\n{text}"
    )
    try:
        resp = llm_tools.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        return f"(Translation failed: {e}) {text}"

translate_tool = Tool(
    name="translate_tool",
    func=translate_text,
    description="Translate a given text into another language. Input: text and optional target_language."
)

# -----------------------------
# Summarize
# -----------------------------
def summarize_text(text: str) -> str:
    """Summarize text into one concise paragraph."""
    prompt = (
        "Summarize the following text into a single concise, fluent paragraph. "
        "Avoid bullet points and keep essential facts.\n\n"
        f"Text:\n{text}"
    )
    try:
        resp = llm_tools.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        return f"(Summary failed: {e}) {text[:300]}..."

summarize_tool = Tool(
    name="summarize_tool",
    func=summarize_text,
    description="Generate a short, fluent paragraph summary of a longer text."
)

# -----------------------------
# Export to PDF
# -----------------------------
def export_to_pdf(text: str, filename: str = "output.pdf") -> str:
    """Generate a PDF file from provided text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, text)
    pdf.output(filename)
    return f"PDF saved as {filename}"

class PdfArgs(BaseModel):
    text: str
    filename: str = "output.pdf"

export_to_pdf_tool = StructuredTool.from_function(
    func=export_to_pdf,
    name="export_to_pdf_tool",
    description="Generate a PDF from provided text and save it to the given filename.",
    args_schema=PdfArgs,
)

# -----------------------------
# Scrape
# -----------------------------
def scrape_page(url: str) -> str:
    """Scrape visible text from a public webpage."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:2000]
    except Exception as e:
        return f"(Scraping failed: {str(e)})"

scrape_tool = Tool(
    name="scrape_tool",
    func=scrape_page,
    description="Scrape visible text content from a public URL (must start with https://)."
)