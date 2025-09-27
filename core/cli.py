"""
Quasar CLI - Interactive command-line interface
"""

import os
import sys
import cmd
import json
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.progress import track
import pandas as pd

from core.agent import QuasarAgent, AgentConfig
from integrations.tap import NRAOTapClient

console = Console()

class QuasarCLI(cmd.Cmd):
    """Interactive CLI for Quasar"""

    intro = """
    ╔══════════════════════════════════════════════════════════╗
    ║  Welcome to Quasar CLI - Radio Astronomy Assistant      ║
    ║  Type 'help' for commands or 'chat' to start chatting   ║
    ║  Type 'exit' or 'quit' to leave                        ║
    ╚══════════════════════════════════════════════════════════╝
    """

    prompt = "[bold cyan]quasar>[/bold cyan] "

    def __init__(self):
        super().__init__()
        self.agent = None
        self.tap_client = None
        self.last_results = None
        self.initialize()

    def initialize(self):
        """Initialize the CLI components"""
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[yellow]Warning: No OpenAI API key found in environment[/yellow]")
            console.print("Chat features will be disabled. Set OPENAI_API_KEY to enable.")
        else:
            try:
                config = AgentConfig(api_key=api_key)
                self.agent = QuasarAgent(config)
                console.print("[green]✓ AI Agent initialized[/green]")
            except Exception as e:
                console.print(f"[red]Failed to initialize agent: {e}[/red]")

        # Initialize TAP client
        try:
            self.tap_client = NRAOTapClient()
            console.print("[green]✓ TAP client connected[/green]")
        except Exception as e:
            console.print(f"[red]Failed to connect to TAP service: {e}[/red]")

    def do_chat(self, arg):
        """Start interactive chat mode: chat"""
        if not self.agent:
            console.print("[red]Chat not available. Please set OPENAI_API_KEY[/red]")
            return

        console.print("[cyan]Entering chat mode. Type 'exit' to return to main menu[/cyan]\n")

        while True:
            try:
                query = Prompt.ask("\n[bold]You")
                if query.lower() in ['exit', 'quit', 'back']:
                    break

                with console.status("[cyan]Thinking...[/cyan]"):
                    response = self.agent.process_query(query)

                console.print("\n[bold green]Quasar:[/bold green]")
                console.print(Markdown(response))

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def do_search(self, arg):
        """Search NRAO archives: search <type> [params]

        Types:
        - position <ra> <dec> [radius]  : Search by coordinates
        - target <name>                  : Search by target name
        - frequency <min_ghz> <max_ghz>  : Search by frequency range
        - project <code>                 : Search by project code

        Examples:
        search position 83.633 22.014 0.5
        search target M31
        search frequency 1.0 2.0
        search project VLA/23A
        """
        if not self.tap_client:
            console.print("[red]TAP client not available[/red]")
            return

        parts = arg.split()
        if len(parts) < 2:
            console.print("[red]Usage: search <type> <params>[/red]")
            return

        search_type = parts[0].lower()

        try:
            with console.status("[cyan]Searching archives...[/cyan]"):
                if search_type == "position":
                    if len(parts) < 3:
                        console.print("[red]Usage: search position <ra> <dec> [radius][/red]")
                        return
                    ra = float(parts[1])
                    dec = float(parts[2])
                    radius = float(parts[3]) if len(parts) > 3 else 0.5
                    results = self.tap_client.cone_search(ra, dec, radius)

                elif search_type == "target":
                    target = " ".join(parts[1:])
                    results = self.tap_client.search_by_target(target)

                elif search_type == "frequency":
                    if len(parts) < 3:
                        console.print("[red]Usage: search frequency <min_ghz> <max_ghz>[/red]")
                        return
                    min_freq = float(parts[1])
                    max_freq = float(parts[2])
                    results = self.tap_client.search_by_frequency(min_freq, max_freq)

                elif search_type == "project":
                    project = " ".join(parts[1:])
                    results = self.tap_client.search_by_project(project)

                else:
                    console.print(f"[red]Unknown search type: {search_type}[/red]")
                    return

            self.display_results(results)
            self.last_results = results

        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")

    def do_query(self, arg):
        """Execute custom ADQL query: query <ADQL>

        Example:
        query SELECT TOP 10 target_name, s_ra, s_dec FROM ivoa.obscore WHERE facility_name='VLA'
        """
        if not self.tap_client:
            console.print("[red]TAP client not available[/red]")
            return

        if not arg:
            console.print("[red]Please provide an ADQL query[/red]")
            return

        try:
            with console.status("[cyan]Executing query...[/cyan]"):
                results = self.tap_client.execute_query(arg)
            self.display_results(results)
            self.last_results = results
        except Exception as e:
            console.print(f"[red]Query failed: {e}[/red]")

    def do_details(self, arg):
        """Get observation details: details <obs_id>"""
        if not self.tap_client:
            console.print("[red]TAP client not available[/red]")
            return

        if not arg:
            console.print("[red]Please provide an observation ID[/red]")
            return

        try:
            with console.status("[cyan]Fetching details...[/cyan]"):
                details = self.tap_client.get_observation_details(arg)

            # Display details in a nice format
            table = Table(title=f"Observation: {arg}", show_header=False)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")

            for key, value in details.items():
                if isinstance(value, dict):
                    table.add_row(key, json.dumps(value, indent=2))
                else:
                    table.add_row(key, str(value))

            console.print(table)

        except Exception as e:
            console.print(f"[red]Failed to get details: {e}[/red]")

    def do_save(self, arg):
        """Save last search results: save [filename]"""
        if self.last_results is None or self.last_results.empty:
            console.print("[yellow]No results to save[/yellow]")
            return

        filename = arg if arg else "quasar_results.csv"

        try:
            self.last_results.to_csv(filename, index=False)
            console.print(f"[green]Results saved to {filename}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to save: {e}[/red]")

    def do_status(self, arg):
        """Show system status"""
        table = Table(title="Quasar Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")

        # Agent status
        agent_status = "Connected" if self.agent else "Not Available"
        agent_details = "Set OPENAI_API_KEY to enable" if not self.agent else "Ready"
        table.add_row("AI Agent", agent_status, agent_details)

        # TAP status
        tap_status = "Connected" if self.tap_client else "Not Connected"
        tap_details = self.tap_client.service_url if self.tap_client else "Connection failed"
        table.add_row("TAP Service", tap_status, tap_details)

        # Results status
        results_status = f"{len(self.last_results)} rows" if self.last_results is not None else "None"
        table.add_row("Last Results", results_status, "")

        console.print(table)

    def do_clear(self, arg):
        """Clear the screen"""
        os.system('clear' if os.name == 'posix' else 'cls')

    def do_reset(self, arg):
        """Reset conversation memory"""
        if self.agent:
            self.agent.reset_conversation()
            console.print("[yellow]Conversation memory cleared[/yellow]")
        else:
            console.print("[red]No active agent to reset[/red]")

    def do_help(self, arg):
        """Show help for commands"""
        if arg:
            # Show help for specific command
            super().do_help(arg)
        else:
            # Show general help
            help_text = """
[bold cyan]Quasar CLI Commands[/bold cyan]

[bold]Chat & AI:[/bold]
  chat              Start interactive chat mode
  reset             Reset conversation memory

[bold]Search Commands:[/bold]
  search            Search NRAO archives (type 'help search' for details)
  query             Execute custom ADQL query
  details           Get detailed observation info

[bold]Data Management:[/bold]
  save              Save last results to CSV

[bold]System:[/bold]
  status            Show system status
  clear             Clear the screen
  help [command]    Show help for command
  exit/quit         Exit Quasar CLI
            """
            console.print(help_text)

    def do_exit(self, arg):
        """Exit the CLI"""
        if Confirm.ask("\n[yellow]Are you sure you want to exit?[/yellow]"):
            console.print("\n[cyan]Thank you for using Quasar. Goodbye![/cyan]")
            return True
        return False

    def do_quit(self, arg):
        """Exit the CLI"""
        return self.do_exit(arg)

    def display_results(self, results: pd.DataFrame):
        """Display search results in a formatted table"""
        if results.empty:
            console.print("[yellow]No results found[/yellow]")
            return

        console.print(f"\n[green]Found {len(results)} observations[/green]\n")

        # Create display table
        table = Table(show_header=True, header_style="bold magenta")

        # Select columns to display
        display_cols = ['target_name', 'facility_name', 'obs_date',
                       'freq_min_ghz', 'freq_max_ghz', 'duration_hours']

        available_cols = [col for col in display_cols if col in results.columns]

        # Add columns
        for col in available_cols:
            table.add_column(col.replace('_', ' ').title())

        # Add rows (limit to first 20)
        for _, row in results.head(20).iterrows():
            row_data = []
            for col in available_cols:
                val = row[col]
                if pd.isna(val):
                    row_data.append("-")
                elif isinstance(val, float):
                    row_data.append(f"{val:.2f}")
                else:
                    row_data.append(str(val))
            table.add_row(*row_data)

        console.print(table)

        if len(results) > 20:
            console.print(f"\n[dim]Showing first 20 of {len(results)} results[/dim]")

        # Summary statistics
        if 'size_gb' in results.columns:
            total_size = results['size_gb'].sum()
            console.print(f"\n[cyan]Total data size: {total_size:.1f} GB[/cyan]")

        if 'duration_hours' in results.columns:
            total_time = results['duration_hours'].sum()
            console.print(f"[cyan]Total observation time: {total_time:.1f} hours[/cyan]")

    def emptyline(self):
        """Handle empty line (do nothing instead of repeating last command)"""
        pass

    def default(self, line):
        """Handle unknown commands"""
        console.print(f"[red]Unknown command: {line}[/red]")
        console.print("Type 'help' for available commands")

    def precmd(self, line):
        """Pre-process commands (for rich prompt)"""
        # Clear any rich console formatting before processing
        return line

    def postcmd(self, stop, line):
        """Post-process commands"""
        return stop

    def cmdloop(self, intro=None):
        """Override cmdloop to use rich console"""
        if intro:
            console.print(intro)

        while True:
            try:
                line = console.input(self.prompt)
                line = self.precmd(line)
                stop = self.onecmd(line)
                stop = self.postcmd(stop, line)
                if stop:
                    break
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' or 'quit' to leave[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    cli = QuasarCLI()
    cli.cmdloop()