#!/usr/bin/env python3
"""
Quasar - Radio Astronomy AI Assistant
Main entry point for the application
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Initialize rich console for beautiful output
console = Console()

# ASCII Art Logo
QUASAR_ASCII = """
     ██████╗ ██╗   ██╗ █████╗ ███████╗ █████╗ ██████╗
    ██╔═══██╗██║   ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗
    ██║   ██║██║   ██║███████║███████╗███████║██████╔╝
    ██║▄▄ ██║██║   ██║██╔══██║╚════██║██╔══██║██╔══██╗
    ╚██████╔╝╚██████╔╝██║  ██║███████║██║  ██║██║  ██║
     ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝
    Radio Astronomy Intelligence System
"""

def display_banner():
    """Display the Quasar banner"""
    banner_text = Text(QUASAR_ASCII, style="bold cyan")
    panel = Panel(
        banner_text,
        border_style="bright_blue",
        padding=(1, 2),
        title="[bold white]Welcome to Quasar[/bold white]",
        subtitle="[italic]v0.1.0[/italic]"
    )
    console.print(panel)

def check_environment() -> bool:
    """Check if required environment variables are set"""
    load_dotenv()

    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        console.print(f"[red]❌ Missing required environment variables:[/red]")
        for var in missing_vars:
            console.print(f"   - {var}")
        console.print("\n[yellow]Please set these in your .env file[/yellow]")
        return False

    optional_vars = ["NASA_ADS_API_KEY"]
    for var in optional_vars:
        if not os.getenv(var):
            console.print(f"[yellow]⚠️  Optional variable {var} not set[/yellow]")

    return True

def launch_web_ui():
    """Launch the Streamlit web interface"""
    console.print("[green]🚀 Launching Quasar Web Interface...[/green]")
    console.print("[cyan]Opening browser at http://localhost:8501[/cyan]\n")

    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])

def launch_cli():
    """Launch the interactive CLI"""
    console.print("[green]🚀 Starting Quasar CLI...[/green]\n")

    try:
        from core.cli import QuasarCLI
        cli = QuasarCLI()
        cli.run()
    except ImportError as e:
        console.print(f"[red]Error: Could not import CLI module: {e}[/red]")
        console.print("[yellow]Make sure all dependencies are installed: pip install -r requirements.txt[/yellow]")
        sys.exit(1)

def run_query(query: str):
    """Run a single query and exit"""
    console.print(f"[cyan]Processing query: {query}[/cyan]\n")

    try:
        from core.agent import QuasarAgent
        agent = QuasarAgent()
        response = agent.process_query(query)
        console.print(response)
    except Exception as e:
        console.print(f"[red]Error processing query: {e}[/red]")
        sys.exit(1)

def test_tap_connection():
    """Test connection to NRAO TAP service"""
    console.print("[cyan]Testing NRAO TAP service connection...[/cyan]")

    try:
        from integrations.tap import NRAOTapClient
        client = NRAOTapClient()

        # Test with a simple query
        result = client.test_connection()
        if result:
            console.print("[green]✅ TAP connection successful![/green]")
            console.print(f"   Service version: {result.get('version', 'Unknown')}")
            console.print(f"   Available tables: {result.get('table_count', 'Unknown')}")
        else:
            console.print("[red]❌ TAP connection failed[/red]")
    except Exception as e:
        console.print(f"[red]❌ Connection error: {e}[/red]")

def list_examples():
    """Display example queries"""
    examples = [
        ("Search by position", "Find VLA observations within 0.5 degrees of M31"),
        ("Search by target", "Show me all Cygnus A observations from 2023"),
        ("Search by frequency", "Find L-band pulsar observations"),
        ("Search by project", "Get data from project VLA/23A-001"),
        ("Complex query", "Find C-band continuum observations of AGN with integration time > 1 hour"),
        ("Download data", "Download raw data for observation uid://A001/X1234/X56"),
        ("Process data", "Calibrate and image the measurement set 'my_data.ms'"),
        ("Visualization", "Create a spectral index map from the last image"),
    ]

    console.print("\n[bold cyan]Example Queries:[/bold cyan]\n")
    for category, query in examples:
        console.print(f"  [bold]{category}:[/bold]")
        console.print(f"    '{query}'\n")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Quasar - Radio Astronomy AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quasar                    # Launch web interface (default)
  quasar cli               # Launch interactive CLI
  quasar query "Find M31"  # Run single query
  quasar test              # Test connections
  quasar examples          # Show example queries
        """
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="web",
        choices=["web", "cli", "query", "test", "examples"],
        help="Launch mode (default: web)"
    )

    parser.add_argument(
        "query_text",
        nargs="*",
        help="Query text (for query mode)"
    )

    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip banner display"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Display banner unless suppressed
    if not args.no_banner:
        display_banner()

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Load custom config if provided
    if args.config:
        console.print(f"[cyan]Loading config from: {args.config}[/cyan]")
        # TODO: Implement config loading

    # Execute based on mode
    try:
        if args.mode == "web":
            launch_web_ui()
        elif args.mode == "cli":
            launch_cli()
        elif args.mode == "query":
            if args.query_text:
                query = " ".join(args.query_text)
                run_query(query)
            else:
                console.print("[red]Error: No query text provided[/red]")
                console.print("Usage: quasar query \"your query here\"")
                sys.exit(1)
        elif args.mode == "test":
            test_tap_connection()
        elif args.mode == "examples":
            list_examples()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        if args.verbose:
            import traceback
            console.print("[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)

if __name__ == "__main__":
    main()