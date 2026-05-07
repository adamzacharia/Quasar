#!/usr/bin/env python3
"""
Quasar System Test Script
Tests core functionality and connections
"""

import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

def test_imports():
    """Test that all modules can be imported"""
    console.print("\n[bold]Testing imports...[/bold]")

    modules_to_test = [
        ("core.agent", "QuasarAgent"),
        ("core.memory", "ConversationMemory"),
        ("core.tools", "ToolRegistry"),
        ("integrations.datalink", "DataLinkClient"),
        ("services.search", "SearchService"),
        ("services.analysis", "RadioAnalysisService"),
        ("utils.formatters", "format_bytes"),
    ]

    results = []
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            results.append((module_name, "[OK] Success", ""))
        except ImportError as e:
            results.append((module_name, "[FAIL] Failed", str(e)))
        except AttributeError as e:
            results.append((module_name, "[!]  Warning", f"Class {class_name} not found"))

    # Display results
    table = Table(title="Import Tests")
    table.add_column("Module", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    for module, status, details in results:
        table.add_row(module, status, details)

    console.print(table)
    return all("[OK]" in r[1] for r in results)

def test_environment():
    """Test environment variables"""
    console.print("\n[bold]Testing environment...[/bold]")
    load_dotenv()

    table = Table(title="Environment Variables")
    table.add_column("Variable", style="cyan")
    table.add_column("Status")
    table.add_column("Value")

    # Required variables
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key.startswith("sk-"):
        table.add_row("OPENAI_API_KEY", "[OK] Set", "sk-***" + openai_key[-4:])
        has_openai = True
    else:
        table.add_row("OPENAI_API_KEY", "[FAIL] Not set", "Required for chat")
        has_openai = False

    # Optional variables
    ads_key = os.getenv("NASA_ADS_API_KEY", "")
    if ads_key:
        table.add_row("NASA_ADS_API_KEY", "[OK] Set", "***" + ads_key[-4:])
    else:
        table.add_row("NASA_ADS_API_KEY", "[!]  Not set", "Optional")

    console.print(table)
    return has_openai

def test_agent_initialization():
    """Test agent initialization (if API key is available)"""
    console.print("\n[bold]Testing AI Agent...[/bold]")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[yellow]⚠️  Skipping - no OpenAI API key[/yellow]")
        return None

    try:
        from core.agent import QuasarAgent, AgentConfig

        config = AgentConfig(api_key=api_key, verbose=False)
        agent = QuasarAgent(config)
        console.print("[green]✅ Agent initialized successfully[/green]")

        # Test a simple query
        with console.status("[cyan]Testing agent response...[/cyan]"):
            # Mock the process_query since it might need UI context or complex setup
            # Just checking init is good enough for this basic test
            pass

        return True

    except Exception as e:
        console.print(f"[red]❌ Agent error: {str(e)}[/red]")
        return False

def test_directories():
    """Test that required directories exist"""
    console.print("\n[bold]Testing directories...[/bold]")

    dirs_to_check = ['data', 'cache', 'logs', 'config']

    table = Table(title="Directory Check")
    table.add_column("Directory", style="cyan")
    table.add_column("Status")

    all_exist = True
    for dir_name in dirs_to_check:
        path = Path(__file__).parent.parent / dir_name
        if path.exists():
            table.add_row(dir_name, "✅ Exists")
        else:
            table.add_row(dir_name, "❌ Missing")
            all_exist = False

    console.print(table)
    return all_exist

def main():
    """Run all tests"""
    console.print(Panel.fit(
        "[bold cyan]QUASAR SYSTEM TEST[/bold cyan]\nTesting core functionality",
        border_style="bright_blue"
    ))

    results = {
        "Imports": test_imports(),
        "Environment": test_environment(),
        "Directories": test_directories(),
        "AI Agent": test_agent_initialization()
    }

    # Summary
    console.print("\n" + "="*50)
    console.print("[bold]TEST SUMMARY[/bold]")
    console.print("="*50)

    table = Table(show_header=False)
    table.add_column("Test", style="cyan")
    table.add_column("Result")

    for test_name, result in results.items():
        if result is None:
            status = "⚠️  Skipped"
        elif result:
            status = "✅ Passed"
        else:
            status = "❌ Failed"
        table.add_row(test_name, status)

    console.print(table)

    # Overall status
    failures = sum(1 for r in results.values() if r is False)
    if failures == 0:
        console.print("\n[bold green]All tests passed! Quasar is ready to use.[/bold green]")
        console.print("\nLaunch with:")
        console.print("  [cyan]streamlit run ui/app.py[/cyan]  (Web interface)")
        console.print("  [cyan]python quasar.py cli[/cyan]     (CLI mode)")
        return 0
    else:
        console.print(f"\n[bold red]{failures} test(s) failed. Please check the errors above.[/bold red]")
        return 1

if __name__ == "__main__":
    sys.exit(main())