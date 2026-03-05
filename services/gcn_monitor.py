"""
GCNAlertMonitor — Gravitational Wave & Transient Alert Monitor for Quasar AI
Parses and summarizes alerts from:
  - NASA GCN (Gamma-ray Coordinates Network) — GW, GRB, neutrino events
  - GWTC (Gravitational Wave Transient Catalog) via GWOSC API

Registered agent tools:
    - get_latest_gcn_alerts(n)
    - search_gwtc_catalog(mass_min, mass_max, event_type)
    - summarize_gcn_circular(circular_id)
"""

import json
from typing import Optional, Dict, Any, List
from datetime import datetime


class GCNAlertMonitor:
    """Parse and query gravitational wave and transient event alerts."""

    GWOSC_API = "https://gwosc.org/eventapi/json/query"
    GCN_CIRCULAR_BASE = "https://gcn.gsfc.nasa.gov/gcn3"

    def get_latest_gcn_alerts(
        self, n: int = 10, event_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the latest GW events from GWOSC (Gravitational Wave Open Science Center).

        Args:
            n: Number of recent events to return (max 50)
            event_type: Filter by type — 'BBH', 'BNS', 'NSBH', 'Burst'

        Returns:
            dict with list of recent GW events and their key parameters
        """
        try:
            import requests

            resp = requests.get(
                self.GWOSC_API,
                params={"event-type": event_type or "", "result": "all"},
                timeout=10
            )

            if resp.status_code != 200:
                return {"success": False, "error": f"GWOSC API returned {resp.status_code}"}

            data = resp.json()
            events = data.get("events", {})

            parsed = []
            for name, info in list(events.items())[:n]:
                params = info.get("parameters", {}) if isinstance(info, dict) else {}
                parsed.append({
                    "name": name,
                    "gps_time": info.get("GPS", "N/A") if isinstance(info, dict) else "N/A",
                    "far": info.get("FAR", "N/A") if isinstance(info, dict) else "N/A",
                    "mass_1_solar": params.get("mass_1_source", {}).get("best", "N/A"),
                    "mass_2_solar": params.get("mass_2_source", {}).get("best", "N/A"),
                    "chirp_mass": params.get("chirp_mass", {}).get("best", "N/A"),
                    "luminosity_distance_mpc": params.get("luminosity_distance", {}).get("best", "N/A"),
                    "network_snr": params.get("network_matched_filter_snr", {}).get("best", "N/A"),
                    "sky_area_90percent_deg2": params.get("sky_area", {}).get("best", "N/A"),
                    "source_type": name[:3] if name[:2] == "GW" else "Unknown",
                    "url": f"https://gwosc.org/eventapi/html/event/{name}/",
                })

            return {
                "success": True,
                "n_events": len(parsed),
                "source": "GWOSC Gravitational Wave Open Science Center",
                "events": parsed,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_gwtc_catalog(
        self,
        mass_min_solar: Optional[float] = None,
        mass_max_solar: Optional[float] = None,
        distance_max_mpc: Optional[float] = None,
        event_type: Optional[str] = None,
        catalog: str = "GWTC-3",
    ) -> Dict[str, Any]:
        """
        Search the Gravitational Wave Transient Catalog (GWTC) with filters.

        Args:
            mass_min_solar: Minimum total mass in solar masses
            mass_max_solar: Maximum total mass in solar masses
            distance_max_mpc: Maximum luminosity distance (Mpc)
            event_type: 'BBH', 'BNS', 'NSBH', or None for all
            catalog: Catalog version (default 'GWTC-3')

        Returns:
            dict with matched event list
        """
        try:
            import requests

            resp = requests.get(self.GWOSC_API, params={"result": "all"}, timeout=10)
            if resp.status_code != 200:
                return {"success": False, "error": f"API status {resp.status_code}"}

            all_events = resp.json().get("events", {})
            matches = []

            for name, info in all_events.items():
                if not isinstance(info, dict):
                    continue
                params = info.get("parameters", {})

                mass1 = params.get("mass_1_source", {}).get("best")
                mass2 = params.get("mass_2_source", {}).get("best")
                dist = params.get("luminosity_distance", {}).get("best")

                total_mass = None
                if mass1 and mass2:
                    try:
                        total_mass = float(mass1) + float(mass2)
                    except Exception:
                        pass

                # Apply filters
                if mass_min_solar is not None and (total_mass is None or total_mass < mass_min_solar):
                    continue
                if mass_max_solar is not None and (total_mass is None or total_mass > mass_max_solar):
                    continue
                if distance_max_mpc is not None and dist is not None:
                    try:
                        if float(dist) > distance_max_mpc:
                            continue
                    except Exception:
                        pass

                matches.append({
                    "name": name,
                    "total_mass_solar": round(total_mass, 1) if total_mass else "N/A",
                    "luminosity_distance_mpc": dist,
                    "far": info.get("FAR", "N/A"),
                    "url": f"https://gwosc.org/eventapi/html/event/{name}/",
                })

            return {
                "success": True,
                "catalog": catalog,
                "filters_applied": {
                    "mass_min": mass_min_solar,
                    "mass_max": mass_max_solar,
                    "distance_max_mpc": distance_max_mpc,
                    "event_type": event_type,
                },
                "n_matches": len(matches),
                "events": matches,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def summarize_gcn_circular(self, circular_number: int) -> Dict[str, Any]:
        """
        Fetch and extract key information from a NASA GCN circular.

        Args:
            circular_number: GCN circular number (e.g., 33000)

        Returns:
            dict with extracted event name, trigger time, coordinates, and summary
        """
        try:
            import requests, re

            url = f"https://gcn.gsfc.nasa.gov/gcn3/{circular_number}.gcn3"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 404:
                return {"success": False, "error": f"GCN Circular #{circular_number} not found."}
            if resp.status_code != 200:
                return {"success": False, "error": f"HTTP {resp.status_code}"}

            text = resp.text

            # Extract key fields with regex
            subject_m = re.search(r"SUBJECT:\s*(.+)", text, re.IGNORECASE)
            from_m = re.search(r"FROM:\s*(.+)", text, re.IGNORECASE)
            date_m = re.search(r"DATE:\s*(.+)", text, re.IGNORECASE)

            # Try to find RA/Dec
            ra_m = re.search(r"RA\s*[=:]\s*([\d.]+)", text)
            dec_m = re.search(r"DEC?\s*[=:]\s*([+-]?[\d.]+)", text)

            return {
                "success": True,
                "circular_number": circular_number,
                "url": url,
                "subject": subject_m.group(1).strip() if subject_m else "Unknown",
                "from": from_m.group(1).strip() if from_m else "Unknown",
                "date": date_m.group(1).strip() if date_m else "Unknown",
                "ra_deg": ra_m.group(1) if ra_m else None,
                "dec_deg": dec_m.group(1) if dec_m else None,
                "body_preview": text[:800].strip(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
