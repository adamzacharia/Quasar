# Quasar

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)

<img width="1897" height="783" alt="image" src="https://github.com/user-attachments/assets/f38936a3-c8f2-4aad-a00a-145327aaf0bb" />

## AI-Powered Radio Astronomy Data Discovery and Processing

Quasar is an intelligent assistant that bridges natural language queries with the National Radio Astronomy Observatory (NRAO) archives, enabling seamless discovery, calibration, and visualization of radio astronomy data.

## Features

- **Natural Language Archive Search** - Query ALMA observations using standard English (powered by `alminer`)
- **Literature Integration** - NASA ADS API for identifying related publications
- **Technical RAG** - Intelligent Q&A based on the ALMA Proposer's Guide
- **AI-Powered Workflow** - Interpret user intent and execute complex queries

## Prerequisites

- Python 3.9+
- OpenAI API key
- NASA ADS API key (optional)


## Installation

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/quasar.git
cd quasar
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Getting Started

### 1. Start the Backend API (Required for UI-Pro)
The backend service powers the core Quasar agent operations and connects to the ALMA Science Archive.
```bash
# From the root quasar directory
conda activate quasar
uvicorn ui-pro.api.main:app --reload --port 8000
```
*The backend will be available at `http://localhost:8000`*

### 2. Start the Frontend (UI-Pro)
The modern web interface provides a chat-like experience for interacting with the Quasar agent.
```bash
# In a new terminal, navigate to the ui-pro directory
cd ui-pro
npm install
npm run dev
```
*Navigate to `http://localhost:3000` in your web browser.*

### Alternative: Streamlit Interface (Legacy)
```bash
streamlit run ui/app.py
```
*Navigate to `http://localhost:8501`*

### Alternative: Python API
```python
from quasar import QuasarAgent

agent = QuasarAgent()
results = agent.search("Find ALMA observations of Sz65 in Band 6")
```

## Usage Examples

### Search for Observations
```
"Show me recent ALMA observations of Sz65"
"Find ALMA data for Lupus I between 2020-2023"
"Search for protoplanetary disks in Band 7"
```

## Architecture

```
quasar/
├── core/           # Core agent logic and LLM integration
├── integrations/   # External service connectors (ALminer, ADS)
├── services/       # Business logic and data processing
├── ui/            # Streamlit web interface
├── utils/         # Helper functions and utilities
└── config/        # Configuration management
```

## Configuration

### Environment Variables (.env)
Create a `.env` file in the root directory:

```ini
# Core API Keys
OPENAI_API_KEY=sk-...

# Optional: For Literature Search
NASA_ADS_API_KEY=...
```

### Settings
Edit `config/settings.py` to customize query limits and default search parameters.

## API Documentation

### QuasarAgent
The main agent class for orchestrating searches.

```python
agent = QuasarAgent(api_key="your-openai-key")
results = agent.search("Find ALMA observations of Sz65", max_results=100)
```

## Troubleshooting Guide

### Common Issues

**ALminer Import Error**
- Ensure `alminer` is installed: `pip install alminer`

**OpenAI API Error**
- Verify your API key in `.env`
- Check your quota/billing status

## License

MIT License - see LICENSE file for details

## Acknowledgements

- National Radio Astronomy Observatory for archive access
- OpenAI for GPT API
- NASA ADS for literature database
- The `alminer` team for their excellent ALMA archive wrapper

## Contact

For questions or support, please open an issue on GitHub.
