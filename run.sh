#!/bin/bash

# Quasar - Radio Astronomy Assistant
# Startup script

echo "========================================="
echo "   🌌 QUASAR - Radio Astronomy Assistant"
echo "========================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "quasar.py" ]; then
    echo "❌ Error: Not in the Quasar directory"
    echo "Please run this script from the Quasar root directory"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Please ensure your environment variables are configured"
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📚 Checking dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt 2>/dev/null || echo "⚠️  Some dependencies may need manual installation"

# Clear terminal for clean start
clear

# Display startup message
echo "╔══════════════════════════════════════════════╗"
echo "║                                              ║"
echo "║     🌌  QUASAR IS STARTING UP  🌌           ║"
echo "║                                              ║"
echo "║     Radio Astronomy Intelligence System      ║"
echo "║                                              ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "🚀 Launching Streamlit interface..."
echo "📡 Opening browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "----------------------------------------"
echo ""

# Run Streamlit app
streamlit run ui/app.py \
    --server.port=8501 \
    --server.address=localhost \
    --browser.serverAddress=localhost \
    --theme.base="dark" \
    --theme.primaryColor="#667eea" \
    --theme.backgroundColor="#0a0e27" \
    --theme.secondaryBackgroundColor="#1a1e3a" \
    --theme.textColor="#e0e0e0"
