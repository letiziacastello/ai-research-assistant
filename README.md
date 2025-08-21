# AI Research Assistant

A conversational AI agent built with **LangChain**, **OpenAI**, and **Streamlit**, designed to assist users in structured online research using external tools such as Wikipedia, DuckDuckGo, translation, summarization, and PDF export.

![AI Research Assistant Screenshot](screenshot_app.png)

---

## Features

-  Interactive chatbot powered by OpenAI's GPT-4o
-  Tool integration with:
  - Wikipedia & DuckDuckGo search
  - Text summarization (via Hugging Face `transformers`)
  - Translation (via `deep_translator`)
  - PDF export of research results
  - Save responses to file
-  Structured output using Pydantic
-  Context memory with LangChain's `ConversationBufferMemory`
- �Beautiful and user-friendly **Streamlit** UI

---

##  Use Case

The user enters a research topic (e.g., *The Renaissance*), and the assistant:
- Searches reliable sources
- Summarizes the content
- Outputs a well-written paragraph
- Shows tools used and saves a copy

---

## Tech Stack

- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [Streamlit](https://streamlit.io/)
- [Transformers](https://huggingface.co/transformers/)
- [Deep Translator](https://pypi.org/project/deep-translator/)
- [FPDF](https://pyfpdf.github.io/)

---

##  Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/letiziacastello/ai-research-assistant.git
cd ai-research-assistant

2. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

3. **Install dependencies**

```bash
pip install -r requirements.txt

4. **Set up environment variables
OPEANAI_API_KEY=your_openai_key_here

5. **Run The Streamlit app
```bash
streamlit run app.py



