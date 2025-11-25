# Quasar 🌌
<img width="1897" height="783" alt="image" src="https://github.com/user-attachments/assets/f38936a3-c8f2-4aad-a00a-145327aaf0bb" />

## AI-Powered Radio Astronomy Data Discovery & Processing

Quasar is an intelligent assistant that bridges natural language queries with the National Radio Astronomy Observatory (NRAO) archives, enabling seamless discovery, calibration, and visualization of radio astronomy data.

## Features

- 🔍 **Natural Language Archive Search** - Query ALMA observations using plain English
- 📚 **Literature Integration** - NASA ADS API for finding related publications
- 📊 **Automated Data Processing** - End-to-end pipeline from raw FITS to calibrated images (coming soon)
- 🎨 **Smart Visualization** - CARTA integration for publication-ready figures (coming soon)
- 🤖 **AI-Powered Workflow** - Interpret intent and execute complex pipeline (coming soon)

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

### 4. Install CASA (Optional, for calibration)
```bash
# Using pip
pip install casatools casatasks casadata

# Or using conda
conda install -c conda-forge casa-core casa-python
```

### 5. Install CARTA (Optional, for visualization)
```bash
# Desktop version
wget https://github.com/CARTAvis/carta/releases/latest/download/carta-backend.tar.gz
tar -xzf carta-backend.tar.gz
```

## Quick Start

### Web Interface
```bash
streamlit run ui/app.py
```
Navigate to http://localhost:8501

### CLI Mode
```bash
python quasar.py cli
```

### Python API
```python
from quasar import QuasarAgent

agent = QuasarAgent()
results = agent.search("Find VLA observations of M31 in L-band")
```

## Usage Examples

### Search for Observations
```
"Show me recent VLA observations of Cygnus A"
"Find ALMA data for NGC 1234 between 2020-2023"
"Search for pulsar observations at 1.4 GHz"
```

### Download and Process
```
"Download the raw data for project VLA/23A-001"
"Calibrate and image the dataset I just downloaded"
"Apply RFI flagging to the L-band data"
```

### Visualization
```
"Create a continuum image with robust weighting"
"Show me the UV coverage for this observation"
"Export a publication-quality figure with contours"
```

## Architecture

```
quasar/
├── core/           # Core agent logic and LLM integration
├── integrations/   # External service connectors (TAP, CASA, CARTA)
├── services/       # Business logic and data processing
├── ui/            # Streamlit web interface
├── utils/         # Helper functions and utilities
└── config/        # Configuration management
```

## Configuration

Edit `config/settings.py` to customize:
- TAP query limits
- Default imaging parameters
- CASA pipeline settings
- CARTA visualization presets

## API Documentation

### QuasarAgent
Main agent class for orchestrating searches and processing.

```python
agent = QuasarAgent(api_key="your-openai-key")
agent.search(query, max_results=100)
agent.download(dataset_id)
agent.process(measurement_set, pipeline="vla_standard")
```

### TAP Integration
Direct access to NRAO's Virtual Observatory TAP service.

```python
from integrations.tap import NRAOTapClient
client = NRAOTapClient()
results = client.cone_search(ra=83.633, dec=22.014, radius=0.5)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

### Common Issues

**ImportError with CASA modules**
- Ensure CASA is properly installed and in PATH
- Try: `export PYTHONPATH=$PYTHONPATH:/usr/local/casa/lib/python3.9`

**TAP queries timeout**
- Reduce query size with `TOP` clause
- Use more specific coordinate constraints

**Memory errors during imaging**
- Reduce image size or use fewer channels
- Increase swap space on WSL2

## License

MIT License - see LICENSE file for details

## Acknowledgments

- National Radio Astronomy Observatory for archive access
- OpenAI for GPT API
- NASA ADS for literature database
- CASA and CARTA development teams

## Contact

For questions or support, please open an issue on GitHub.

---

*"Exploring the radio universe, one query at a time"* 🔭


