#!/bin/bash

# Quasar Setup Script
# Quick setup for the Quasar radio astronomy AI assistant

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════╗"
echo "║       QUASAR SETUP SCRIPT                ║"
echo "║   Radio Astronomy Intelligence System    ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python 3.9+ is required (found $python_version)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version${NC}"

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install requirements
echo "Installing Python packages..."
echo "This may take a few minutes..."
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓ Python packages installed${NC}"

# Create necessary directories
echo "Creating project directories..."
mkdir -p data cache logs config
echo -e "${GREEN}✓ Directories created${NC}"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env and add your API keys${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Optional: Install CASA (commented out by default due to complexity)
# echo "Installing CASA (optional)..."
# pip install casatools casatasks casadata

# Test imports
echo "Testing imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from core.agent import QuasarAgent
    from integrations.tap import NRAOTapClient
    print('✓ Core modules imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

# Test TAP connection
echo "Testing NRAO TAP connection..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from integrations.tap import NRAOTapClient
    client = NRAOTapClient()
    result = client.test_connection()
    if result['status'] == 'connected':
        print('✓ TAP connection successful')
    else:
        print('✗ TAP connection failed')
except Exception as e:
    print(f'✗ Connection error: {e}')
"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key:"
echo "   ${YELLOW}code .env${NC}"
echo ""
echo "2. Launch Quasar:"
echo "   - Web interface: ${GREEN}streamlit run ui/app.py${NC}"
echo "   - CLI mode: ${GREEN}python quasar.py cli${NC}"
echo "   - Quick query: ${GREEN}python quasar.py query \"Find VLA observations of M31\"${NC}"
echo ""
echo "3. Test the system:"
echo "   ${GREEN}python quasar.py test${NC}"
echo ""
echo "For help: ${GREEN}python quasar.py --help${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"

# Reminder about activation
echo ""
echo -e "${YELLOW}Remember to activate the virtual environment:${NC}"
echo "  source venv/bin/activate"
echo ""

# Check for API key
if ! grep -q "sk-" .env 2>/dev/null; then
    echo -e "${RED}⚠️  Don't forget to add your OpenAI API key to .env!${NC}"
fi