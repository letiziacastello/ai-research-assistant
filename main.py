import os
import sys
import argparse
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory

from tools import (
    search_tool,
    wiki_tool,
    save_tool,
    translate_tool,
    summarize_tool,
    export_to_pdf_tool,
    scrape_tool,
)

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Missing OPENAI_API_KEY in environment. Create a .env with OPENAI_API_KEY=...")
    sys.exit(1)

# -------------------------
# Define the structured output schema
# -------------------------
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[str]
    tools_used: List[str]

# -------------------------
# LLM & output parser
# -------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
format_instructions = parser.get_format_instructions()

# -------------------------
# System prompt
# -------------------------
SYSTEM_TEXT = f"""
# ROLE:
You are a highly capable AI Research Assistant.
You specialize in generating structured research outputs and can use external tools when necessary.

# TASK:
- Help the user research a given topic.
- Gather, synthesize, and present the information in a single, fluent paragraph (no bullet points).
- Ensure clarity, coherence, and conciseness.

# OUTPUT:
Return ONLY a JSON object matching this schema:

{format_instructions}

# TOOLS AVAILABLE:
- search (DuckDuckGo), wikipedia, summarize, translate, scrape, export to PDF, save to file.

# RULES:
- Use tools only when necessary.
- Do not fabricate sources; list only sources you actually used.
- If translation is requested, perform it as the final step.

# EXAMPLES:
User: "Research climate change and translate it in Italian"
Assistant: Research → Summarize → Translate → Return JSON.

User: "Tell me about blockchain and summarize it"
Assistant: Research → Summarize → Return JSON.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEXT),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# -------------------------
# Tools & agent creation
# -------------------------
tools = [
    search_tool,
    wiki_tool,
    save_tool,
    translate_tool,
    summarize_tool,
    export_to_pdf_tool,
    scrape_tool,
]

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,                 # Set to False if you don't want logs
    return_intermediate_steps=False,  # True if you want step-by-step debug
    return_only_outputs=True,
)

# -------------------------
# Command Line Interface
# -------------------------
def run_cli():
    ap = argparse.ArgumentParser(description="Research Agent (CLI)")
    ap.add_argument("query", nargs="*", help="Your research query")
    ap.add_argument("--translate", "-t", default="", help="Translate the final summary to this language (e.g., Italian)")
    args = ap.parse_args()

    # Get query either from CLI args or user input
    user_query = " ".join(args.query).strip()
    if not user_query:
        user_query = input("What can I help you research? ").strip()

    # Append translation request if provided
    if args.translate:
        user_query += f"\n\nPlease translate the final summary into {args.translate}."

    try:
        raw = executor.invoke({"query": user_query})
        output_text = raw.get("output", raw)
        data = parser.parse(output_text)

        print("\n Parsed JSON Output:")
        print(data.model_dump_json(indent=2, ensure_ascii=False))

    except ValidationError as ve:
        print("Pydantic validation error:")
        print(ve)
        sys.exit(2)

    except Exception as e:
        print("Agent execution error:")
        print(e)
        if isinstance(raw, dict):
            print("\nRaw output:\n", raw.get("output", raw))
        sys.exit(3)

if __name__ == "__main__":
    run_cli()