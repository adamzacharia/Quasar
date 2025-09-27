"""
QuasarAgent - Core AI agent for radio astronomy operations
Handles natural language processing and tool orchestration
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import openai
from openai import OpenAI
from rich.console import Console

from core.memory import ConversationMemory
from core.tools import ToolRegistry, Tool
from integrations.tap import NRAOTapClient
from integrations.datalink import DataLinkClient
from services.search import SearchService
from services.analysis import RadioAnalysisService

console = Console()

@dataclass
class AgentConfig:
    """Configuration for QuasarAgent"""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = "gpt-4o-mini"  # Changed from gpt-4-turbo-preview to gpt-4o-mini
    temperature: float = 0.7
    max_tokens: int = 2000
    max_memory_turns: int = 10
    verbose: bool = False

class QuasarAgent:
    """Main AI agent for radio astronomy operations"""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Quasar agent"""
        self.config = config or AgentConfig()

        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.config.api_key)

        # Initialize components
        self.memory = ConversationMemory(max_turns=self.config.max_memory_turns)
        self.tool_registry = ToolRegistry()
        self.tap_client = NRAOTapClient()
        self.search_service = SearchService(self.tap_client)
        self.analysis_service = RadioAnalysisService()

        # Register tools
        self._register_tools()

        # System prompt
        self.system_prompt = self._build_system_prompt()

        if self.config.verbose:
            console.print("[green]QuasarAgent initialized successfully[/green]")

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent"""
        return """You are Quasar, an expert AI assistant for radio astronomy data discovery and analysis.

Your capabilities include:
1. Searching NRAO archives (VLA, VLBA, ALMA, GBT) using natural language
2. Converting queries to ADQL/TAP searches for metadata discovery
3. Providing information about radio astronomy observations (metadata only)
4. Analyzing observation parameters and coverage
5. Creating visualizations of search results
6. Providing expert guidance on radio astronomy techniques

IMPORTANT LIMITATIONS:
- TAP services provide observation metadata only, not actual data files
- For data downloads, users must use official archive portals:
  * NRAO Data Portal: https://data.nrao.edu/
  * ALMA Science Archive: https://almascience.nrao.edu/aq/
  * VLA/VLBA Archive: https://archive.nrao.edu/

When responding:
- Be concise but thorough
- Provide specific examples when helpful
- Explain technical concepts clearly
- Suggest follow-up actions when appropriate
- Format numerical results in tables when possible
- Always clarify that you provide metadata discovery, not data downloads

Current date: {date}
Available tools: {tools}
""".format(
            date=datetime.now().strftime("%Y-%m-%d"),
            tools=", ".join([t.name for t in self.tool_registry.list_tools()])
        )

    def _register_tools(self):
        """Register available tools with the agent"""

        # Search tools
        self.tool_registry.register(Tool(
            name="search_by_position",
            description="Search NRAO archives by sky position (cone search)",
            function=self._search_by_position,
            parameters={
                "ra": "Right ascension in degrees",
                "dec": "Declination in degrees",
                "radius": "Search radius in degrees",
                "facility": "Optional: VLA, VLBA, ALMA, or GBT",
                "max_results": "Maximum results to return (default 100)"
            }
        ))

        self.tool_registry.register(Tool(
            name="search_by_target",
            description="Search NRAO archives by target name",
            function=self._search_by_target,
            parameters={
                "target_name": "Name of the astronomical target",
                "facility": "Optional: VLA, VLBA, ALMA, or GBT",
                "date_range": "Optional: date range as 'YYYY-MM-DD,YYYY-MM-DD'",
                "max_results": "Maximum results to return"
            }
        ))

        self.tool_registry.register(Tool(
            name="search_by_frequency",
            description="Search archives by frequency range",
            function=self._search_by_frequency,
            parameters={
                "min_freq_ghz": "Minimum frequency in GHz",
                "max_freq_ghz": "Maximum frequency in GHz",
                "facility": "Optional: VLA, VLBA, ALMA, or GBT",
                "max_results": "Maximum results to return"
            }
        ))

        self.tool_registry.register(Tool(
            name="get_observation_details",
            description="Get detailed information about a specific observation",
            function=self._get_observation_details,
            parameters={
                "obs_id": "Observation ID or execution block ID"
            }
        ))

        self.tool_registry.register(Tool(
            name="download_data",
            description="Download raw data for an observation",
            function=self._download_data,
            parameters={
                "obs_id": "Observation ID",
                "output_dir": "Output directory (default: ./data)"
            }
        ))

        self.tool_registry.register(Tool(
            name="analyze_uv_coverage",
            description="Analyze UV coverage for an observation",
            function=self._analyze_uv_coverage,
            parameters={
                "ms_path": "Path to measurement set"
            }
        ))

    def _search_by_position(self, ra: float, dec: float, radius: float = 0.5,
                           facility: Optional[str] = None,
                           max_results: int = 100) -> Dict[str, Any]:
        """Search archives by sky position"""
        try:
            results = self.search_service.cone_search(
                ra, dec, radius, facility, max_results
            )
            return {
                "success": True,
                "count": len(results),
                "results": results.to_dict("records") if not results.empty else []
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _search_by_target(self, target_name: str, facility: Optional[str] = None,
                          date_range: Optional[str] = None,
                          max_results: int = 100) -> Dict[str, Any]:
        """Search archives by target name"""
        try:
            results = self.search_service.search_by_target(
                target_name, facility, date_range, max_results
            )
            return {
                "success": True,
                "count": len(results),
                "results": results.to_dict("records") if not results.empty else []
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _search_by_frequency(self, min_freq_ghz: float, max_freq_ghz: float,
                            facility: Optional[str] = None,
                            max_results: int = 100) -> Dict[str, Any]:
        """Search archives by frequency range"""
        try:
            results = self.search_service.search_by_frequency(
                min_freq_ghz, max_freq_ghz, facility, max_results
            )
            return {
                "success": True,
                "count": len(results),
                "results": results.to_dict("records") if not results.empty else []
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_observation_details(self, obs_id: str) -> Dict[str, Any]:
        """Get detailed observation information"""
        try:
            details = self.search_service.get_observation_details(obs_id)
            return {"success": True, "details": details}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _download_data(self, obs_id: str, output_dir: str = "./data") -> Dict[str, Any]:
        """Download observation data - Note: TAP only provides metadata, not data files"""
        try:
            # TAP services only provide metadata, not actual data files
            # For actual data download, you would need to use:
            # 1. NRAO Data Portal web interface
            # 2. DataLink services (if available)
            # 3. Direct archive access APIs

            return {
                "success": False,
                "error": "TAP services only provide observation metadata, not data files. "
                        "To download actual data, please use:\n"
                        "1. NRAO Data Portal: https://data.nrao.edu/\n"
                        "2. ALMA Science Archive: https://almascience.nrao.edu/aq/\n"
                        "3. VLA/VLBA Archive: https://archive.nrao.edu/",
                "obs_id": obs_id,
                "suggested_action": "Use observation ID to search for data in the appropriate archive portal"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _analyze_uv_coverage(self, ms_path: str) -> Dict[str, Any]:
        """Analyze UV coverage of a measurement set"""
        try:
            analysis = self.analysis_service.analyze_uv_coverage(ms_path)
            return {"success": True, "analysis": analysis}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_tool_calls(self, response: str) -> List[Tuple[str, Dict]]:
        """Extract tool calls from the LLM response"""
        tool_calls = []

        # Look for tool call patterns in the response
        # Format: [TOOL: tool_name {params}]
        import re
        pattern = r'\[TOOL:\s*(\w+)\s*({.*?})\]'
        matches = re.finditer(pattern, response, re.DOTALL)

        for match in matches:
            tool_name = match.group(1)
            try:
                params = json.loads(match.group(2))
                tool_calls.append((tool_name, params))
            except json.JSONDecodeError:
                if self.config.verbose:
                    console.print(f"[yellow]Failed to parse parameters for {tool_name}[/yellow]")

        return tool_calls

    def process_query(self, query: str) -> str:
        """Process a user query and return response with real data"""
        self.memory.add_message("user", query)

        # Check for search intent
        query_lower = query.lower()

        # Direct search execution for common patterns
        if "find" in query_lower or "search" in query_lower or "show" in query_lower:
            response_text = ""

            # Extract search parameters from natural language
            if "vla" in query_lower:
                facility = "VLA"
            elif "vlba" in query_lower:
                facility = "VLBA"
            elif "alma" in query_lower:
                facility = "ALMA"
            else:
                facility = None

            # Look for target names
            if "m31" in query_lower or "andromeda" in query_lower:
                # Direct target search
                response_text = f"Searching NRAO archives for M31 observations"
                if facility:
                    response_text += f" from {facility}"
                response_text += "...\n\n"

                try:
                    results = self.search_service.search_by_target(
                        "M31", facility=facility, max_results=50
                    )

                    if not results.empty:
                        response_text += f"**Found {len(results)} observations of M31:**\n\n"

                        # Group by facility
                        if 'facility_name' in results.columns:
                            facility_counts = results['facility_name'].value_counts()
                            response_text += "**By Facility:**\n"
                            for fac, count in facility_counts.items():
                                response_text += f"- {fac}: {count} observations\n"

                        response_text += "\n**Recent Observations:**\n\n"
                        # Show table of recent observations
                        recent = results.head(10)

                        # Format as markdown table
                        response_text += "| Date | Facility | Frequency (GHz) | Duration (hrs) |\n"
                        response_text += "|------|----------|----------------|----------------|\n"

                        for _, row in recent.iterrows():
                            date = row.get('obs_date', 'N/A')
                            if pd.notna(date):
                                date = pd.to_datetime(date).strftime('%Y-%m-%d')

                            facility = row.get('facility_name', 'N/A')

                            freq_min = row.get('freq_min_ghz', 0)
                            freq_max = row.get('freq_max_ghz', 0)
                            if freq_min and freq_max:
                                freq = f"{freq_min:.1f}-{freq_max:.1f}"
                            else:
                                freq = "N/A"

                            duration = row.get('duration_hours', 0)
                            if duration:
                                duration = f"{duration:.1f}"
                            else:
                                duration = "N/A"

                            response_text += f"| {date} | {facility} | {freq} | {duration} |\n"

                        # Summary statistics
                        if 'duration_hours' in results.columns:
                            total_time = results['duration_hours'].sum()
                            response_text += f"\n**Total observation time:** {total_time:.1f} hours\n"

                        if 'size_gb' in results.columns:
                            total_size = results['size_gb'].sum()
                            response_text += f"**Total data volume:** {total_size:.1f} GB\n"

                    else:
                        response_text += "No observations found for M31.\n"

                except Exception as e:
                    response_text += f"Search failed: {str(e)}\n"

            # Handle position searches
            elif "cygnus" in query_lower or "3c" in query_lower or any(word in query_lower for word in ["near", "around", "position"]):
                # Extract coordinates if possible
                # For known sources, resolve coordinates
                target = None
                if "cygnus a" in query_lower:
                    target = "Cygnus A"
                    ra, dec = 299.868, 40.734  # Cygnus A coordinates
                elif "3c273" in query_lower:
                    target = "3C273"
                    ra, dec = 187.278, 2.052
                else:
                    # Try to extract coordinates from query
                    import re
                    coord_pattern = r'(\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)'
                    match = re.search(coord_pattern, query)
                    if match:
                        ra = float(match.group(1))
                        dec = float(match.group(2))
                        target = f"Position ({ra:.3f}, {dec:.3f})"
                    else:
                        response_text = "Please specify coordinates or a target name.\n"
                        self.memory.add_message("assistant", response_text)
                        return response_text

                response_text = f"Searching for observations near {target}...\n\n"

                try:
                    results = self.search_service.cone_search(
                        ra, dec, radius=0.5, facility=facility, max_results=50
                    )

                    if not results.empty:
                        response_text += f"**Found {len(results)} observations:**\n\n"

                        # Group by facility
                        if 'facility_name' in results.columns:
                            facility_counts = results['facility_name'].value_counts()
                            response_text += "**By Facility:**\n"
                            for fac, count in facility_counts.items():
                                response_text += f"- {fac}: {count} observations\n"

                        response_text += "\n**Recent Observations:**\n\n"
                        # Show table of recent observations
                        recent = results.head(10)

                        # Format as markdown table
                        response_text += "| Date | Facility | Target | Frequency (GHz) | Duration (hrs) |\n"
                        response_text += "|------|----------|--------|----------------|----------------|\n"

                        for _, row in recent.iterrows():
                            date = row.get('obs_date', 'N/A')
                            if pd.notna(date):
                                date = pd.to_datetime(date).strftime('%Y-%m-%d')

                            facility = row.get('facility_name', 'N/A')
                            target = row.get('target_name', 'N/A')

                            freq_min = row.get('freq_min_ghz', 0)
                            freq_max = row.get('freq_max_ghz', 0)
                            if freq_min and freq_max:
                                freq = f"{freq_min:.1f}-{freq_max:.1f}"
                            else:
                                freq = "N/A"

                            duration = row.get('duration_hours', 0)
                            if duration:
                                duration = f"{duration:.1f}"
                            else:
                                duration = "N/A"

                            response_text += f"| {date} | {facility} | {target} | {freq} | {duration} |\n"

                        # Summary statistics
                        if 'duration_hours' in results.columns:
                            total_time = results['duration_hours'].sum()
                            response_text += f"\n**Total observation time:** {total_time:.1f} hours\n"

                        if 'size_gb' in results.columns:
                            total_size = results['size_gb'].sum()
                            response_text += f"**Total data volume:** {total_size:.1f} GB\n"

                    else:
                        response_text += f"No observations found near {target}.\n"

                except Exception as e:
                    response_text += f"Search failed: {str(e)}\n"

            # Handle frequency searches
            elif "ghz" in query_lower or "band" in query_lower or "frequency" in query_lower:
                # Extract frequency range
                import re
                freq_pattern = r'(\d+\.?\d*)\s*(?:to|-)\s*(\d+\.?\d*)\s*ghz'
                match = re.search(freq_pattern, query_lower)

                if match:
                    min_freq = float(match.group(1))
                    max_freq = float(match.group(2))
                    response_text = f"Searching for observations from {min_freq} to {max_freq} GHz...\n\n"

                    try:
                        results = self.search_service.search_by_frequency(
                            min_freq, max_freq, facility=facility, max_results=50
                        )

                        if not results.empty:
                            response_text += f"**Found {len(results)} observations in this frequency range:**\n\n"

                            # Group by facility
                            if 'facility_name' in results.columns:
                                facility_counts = results['facility_name'].value_counts()
                                response_text += "**By Facility:**\n"
                                for fac, count in facility_counts.items():
                                    response_text += f"- {fac}: {count} observations\n"

                            response_text += "\n**Recent Observations:**\n\n"
                            # Show table of recent observations
                            recent = results.head(10)

                            # Format as markdown table
                            response_text += "| Date | Facility | Target | Frequency (GHz) | Duration (hrs) |\n"
                            response_text += "|------|----------|--------|----------------|----------------|\n"

                            for _, row in recent.iterrows():
                                date = row.get('obs_date', 'N/A')
                                if pd.notna(date):
                                    date = pd.to_datetime(date).strftime('%Y-%m-%d')

                                facility = row.get('facility_name', 'N/A')
                                target = row.get('target_name', 'N/A')

                                freq_min = row.get('freq_min_ghz', 0)
                                freq_max = row.get('freq_max_ghz', 0)
                                if freq_min and freq_max:
                                    freq = f"{freq_min:.1f}-{freq_max:.1f}"
                                else:
                                    freq = "N/A"

                                duration = row.get('duration_hours', 0)
                                if duration:
                                    duration = f"{duration:.1f}"
                                else:
                                    duration = "N/A"

                                response_text += f"| {date} | {facility} | {target} | {freq} | {duration} |\n"

                            # Summary statistics
                            if 'duration_hours' in results.columns:
                                total_time = results['duration_hours'].sum()
                                response_text += f"\n**Total observation time:** {total_time:.1f} hours\n"

                            if 'size_gb' in results.columns:
                                total_size = results['size_gb'].sum()
                                response_text += f"**Total data volume:** {total_size:.1f} GB\n"

                        else:
                            response_text += f"No observations found in the {min_freq}-{max_freq} GHz range.\n"

                    except Exception as e:
                        response_text += f"Search failed: {str(e)}\n"

            self.memory.add_message("assistant", response_text)
            return response_text

        # For non-search queries, use the LLM
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        for msg in self.memory.get_history():
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            assistant_response = response.choices[0].message.content
            self.memory.add_message("assistant", assistant_response)
            return assistant_response

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return error_msg

    def reset_conversation(self):
        """Reset the conversation memory"""
        self.memory.clear()
        if self.config.verbose:
            console.print("[yellow]Conversation memory cleared[/yellow]")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        return self.memory.get_history()

    def set_model(self, model: str):
        """Change the LLM model"""
        self.config.model = model
        if self.config.verbose:
            console.print(f"[cyan]Model changed to: {model}[/cyan]")