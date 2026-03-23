# Lex Fridman Podcast RAG Bot

This project is a Retrieval-Augmented Generation (RAG) bot that allows you to chat with the transcripts of Lex Fridman's podcast episodes.

## 🚀 Features

- **Automated Ingestion**: Downloads transcripts directly from YouTube, chunks them, and builds a searchable FAISS vector index.
- **Local Embeddings**: Uses `fastembed` for efficient, local vector generation.
- **AI-Powered Answers**: Uses Google Gemini to generate answers grounded strictly in the podcast transcript.
- **Rich CLI**: A beautiful terminal interface with clickable source links and formatted tables.

## 🛠️ Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/EdgarTad2002/lex-fridman-rag-bot.git
   cd lex-fridman-rag-bot
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**:
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```text
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## 📖 Usage

### 1. Ingest a Podcast
Run the ingestion script to download and index a specific YouTube video (default is Lex Fridman + Elon Musk):
```bash
python3 ingest.py
```
*Note: This will create a `data/` folder containing the index and chunks.*

### 2. Chat with the Bot
Start the interactive CLI:
```bash
python3 chat.py
```

## 📄 License
MIT
