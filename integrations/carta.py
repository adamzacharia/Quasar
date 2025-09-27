"""
CARTA Integration Module
Provides interface to CARTA for visualization
CARTA (Cube Analysis and Rendering Tool for Astronomy) is a visualization tool
"""

import os
import subprocess
import json
import time
import tempfile
import socket
import webbrowser
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime

# For HTTP requests to CARTA backend
import requests
import asyncio
import websocket

class CARTAIntegration:
    """CARTA integration for radio data visualization"""

    def __init__(self, backend_port: int = 3002, frontend_port: int = 3000):
        """
        Initialize CARTA integration

        Args:
            backend_port: Port for CARTA backend
            frontend_port: Port for CARTA frontend
        """
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        self.base_url = f"http://localhost:{backend_port}"
        self.frontend_url = f"http://localhost:{frontend_port}"
        self.process = None
        self.session_id = None
        self.carta_executable = self._find_carta()

    def _find_carta(self) -> Optional[str]:
        """Find CARTA executable"""
        # Check common locations
        possible_paths = [
            "/usr/local/bin/carta",
            "/usr/bin/carta",
            os.path.expanduser("~/carta/carta"),
            os.path.expanduser("~/carta-backend/carta"),
            "./carta/carta",
            "carta"  # In PATH
        ]

        for path in possible_paths:
            if Path(path).exists() or self._command_exists(path):
                return path

        return None

    def _command_exists(self, command: str) -> bool:
        """Check if command exists in PATH"""
        try:
            subprocess.run(["which", command],
                         capture_output=True,
                         check=True)
            return True
        except:
            return False

    def _is_port_open(self, port: int) -> bool:
        """Check if port is available"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0

    def start_carta_session(self, data_directory: str = "./data",
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start CARTA backend session

        Args:
            data_directory: Directory containing data files
            config: CARTA configuration

        Returns:
            Session information
        """
        if not self.carta_executable:
            return {
                "status": "error",
                "message": "CARTA not found. Please install CARTA.",
                "install_instructions": self._get_install_instructions()
            }

        # Check if already running
        if self._is_port_open(self.backend_port):
            return {
                "status": "already_running",
                "url": self.frontend_url,
                "message": f"CARTA may already be running on port {self.backend_port}"
            }

        try:
            # Prepare command
            cmd = [
                self.carta_executable,
                "--port", str(self.backend_port),
                "--top_level_folder", str(Path(data_directory).absolute()),
                "--no_browser"
            ]

            if config:
                if config.get("no_log"):
                    cmd.append("--no_log")
                if config.get("debug"):
                    cmd.append("--debug")

            # Start CARTA backend
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for startup
            time.sleep(3)

            # Check if running
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                return {
                    "status": "error",
                    "message": "CARTA failed to start",
                    "error": stderr.decode()
                }

            self.session_id = f"session_{int(time.time())}"

            return {
                "status": "success",
                "session_id": self.session_id,
                "backend_port": self.backend_port,
                "frontend_url": self.frontend_url,
                "message": f"CARTA started. Open browser to {self.frontend_url}"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to start CARTA: {str(e)}"
            }

    def stop_carta_session(self) -> Dict[str, Any]:
        """Stop CARTA backend session"""
        if self.process:
            try:
                self.process.terminate()
                time.sleep(1)
                if self.process.poll() is None:
                    self.process.kill()
                self.process = None
                self.session_id = None

                return {
                    "status": "success",
                    "message": "CARTA session stopped"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to stop CARTA: {str(e)}"
                }
        else:
            return {
                "status": "info",
                "message": "No CARTA session running"
            }

    def open_image(self, image_path: str,
                  session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Open image in CARTA

        Args:
            image_path: Path to image file
            session_id: CARTA session ID

        Returns:
            Operation result
        """
        if not self._is_port_open(self.backend_port):
            return {
                "status": "error",
                "message": "CARTA backend not running. Start a session first."
            }

        # CARTA typically auto-loads files in the data directory
        # For full integration, we'd need WebSocket connection

        return {
            "status": "info",
            "message": f"Image {image_path} available in CARTA file browser",
            "instructions": [
                f"1. Open {self.frontend_url} in your browser",
                f"2. Click 'File' -> 'Open Image'",
                f"3. Navigate to {image_path}",
                "4. Click 'Open'"
            ]
        }

    def generate_carta_config(self, config_type: str = "continuum") -> Dict[str, Any]:
        """
        Generate CARTA configuration for different observation types

        Args:
            config_type: Type of configuration (continuum, spectral, polarization)

        Returns:
            Configuration dictionary
        """
        configs = {
            "continuum": {
                "render": {
                    "scaling": "log",
                    "colormap": "viridis",
                    "percentile_min": 0.5,
                    "percentile_max": 99.5,
                    "nan_color": "#000000"
                },
                "contours": {
                    "enabled": True,
                    "levels": [-3, 3, 6, 12, 24, 48, 96],
                    "smoothing": 1,
                    "color": "#ffffff"
                },
                "grid": {
                    "enabled": True,
                    "coordinate_system": "J2000",
                    "grid_type": "automatic"
                }
            },
            "spectral": {
                "render": {
                    "scaling": "linear",
                    "colormap": "rainbow",
                    "percentile_min": 2.0,
                    "percentile_max": 98.0
                },
                "moment": {
                    "types": ["integrated", "velocity", "dispersion"],
                    "velocity_range": "auto"
                },
                "profile": {
                    "enabled": True,
                    "region": "cursor",
                    "statistic": "mean"
                }
            },
            "polarization": {
                "render": {
                    "scaling": "linear",
                    "colormap": "seismic",
                    "percentile_min": 0.0,
                    "percentile_max": 100.0
                },
                "vectors": {
                    "enabled": True,
                    "type": "polarization_angle",
                    "thickness": 1,
                    "spacing": 10,
                    "threshold": 3.0
                },
                "stokes": {
                    "parameters": ["I", "Q", "U", "V"],
                    "display": "I"
                }
            },
            "time_series": {
                "animation": {
                    "enabled": True,
                    "frame_rate": 5,
                    "loop": True
                },
                "differencing": {
                    "enabled": False,
                    "reference_frame": 0
                }
            }
        }

        config = configs.get(config_type, configs["continuum"])
        config["type"] = config_type
        config["timestamp"] = datetime.now().isoformat()

        return config

    def export_publication_quality(self, output_file: str,
                                  width_inches: float = 6.0,
                                  dpi: int = 300,
                                  format: str = "png") -> Dict[str, Any]:
        """
        Export publication-quality figure

        Args:
            output_file: Output filename
            width_inches: Figure width in inches
            dpi: Dots per inch
            format: Output format (png, pdf, svg)

        Returns:
            Export result
        """
        # This would require WebSocket connection to CARTA
        # For now, provide instructions

        export_config = {
            "filename": output_file,
            "format": format,
            "quality": {
                "width_inches": width_inches,
                "height_inches": width_inches * 0.8,  # Aspect ratio
                "dpi": dpi
            },
            "elements": {
                "colorbar": True,
                "coordinate_grid": True,
                "beam": True,
                "title": True,
                "labels": True
            }
        }

        return {
            "status": "config_generated",
            "config": export_config,
            "instructions": [
                "In CARTA:",
                "1. File -> Export Image",
                f"2. Set width to {width_inches} inches",
                f"3. Set DPI to {dpi}",
                f"4. Select format: {format}",
                f"5. Save as: {output_file}"
            ]
        }

    def create_rgb_composite(self, red_image: str, green_image: str,
                           blue_image: str, output: str = "composite.png") -> Dict[str, Any]:
        """
        Create RGB composite instructions

        Args:
            red_image: Path to red channel image
            green_image: Path to green channel image
            blue_image: Path to blue channel image
            output: Output filename

        Returns:
            Composite instructions
        """
        return {
            "status": "instructions",
            "output": output,
            "steps": [
                "In CARTA:",
                f"1. Open {red_image} as base image",
                "2. File -> Append Image",
                f"3. Select {green_image} and {blue_image}",
                "4. View -> RGB Composite",
                f"5. Assign Red: {Path(red_image).name}",
                f"6. Assign Green: {Path(green_image).name}",
                f"7. Assign Blue: {Path(blue_image).name}",
                "8. Adjust scaling for each channel",
                f"9. Export as {output}"
            ]
        }

    def generate_analysis_regions(self, region_type: str = "source") -> List[Dict[str, Any]]:
        """
        Generate region definitions for analysis

        Args:
            region_type: Type of regions (source, background, etc.)

        Returns:
            List of region definitions
        """
        regions = []

        if region_type == "source":
            regions = [
                {
                    "type": "circle",
                    "name": "source_aperture",
                    "center": "image_center",
                    "radius": "3_beam_widths",
                    "color": "#00ff00"
                },
                {
                    "type": "annulus",
                    "name": "background",
                    "inner_radius": "5_beam_widths",
                    "outer_radius": "8_beam_widths",
                    "color": "#ff0000"
                }
            ]
        elif region_type == "spectral":
            regions = [
                {
                    "type": "rectangle",
                    "name": "line_region",
                    "velocity_range": [-100, 100],  # km/s
                    "color": "#0000ff"
                },
                {
                    "type": "rectangle",
                    "name": "continuum_region",
                    "velocity_range": [150, 300],
                    "color": "#ffff00"
                }
            ]
        elif region_type == "mosaic":
            # Grid of regions for mosaic analysis
            for i in range(3):
                for j in range(3):
                    regions.append({
                        "type": "rectangle",
                        "name": f"tile_{i}_{j}",
                        "grid_position": [i, j],
                        "color": "#ffffff"
                    })

        return regions

    def _get_install_instructions(self) -> Dict[str, Any]:
        """Get CARTA installation instructions"""
        return {
            "ubuntu_wsl": [
                "# Download CARTA AppImage",
                "wget https://github.com/CARTAvis/carta/releases/latest/download/CARTA.AppImage",
                "chmod +x CARTA.AppImage",
                "./CARTA.AppImage --appimage-extract",
                "sudo mv squashfs-root /opt/carta",
                "sudo ln -s /opt/carta/AppRun /usr/local/bin/carta"
            ],
            "conda": [
                "conda install -c carta carta-backend"
            ],
            "docker": [
                "docker pull cartavis/carta",
                "docker run -p 3002:3002 -v $PWD:/data cartavis/carta"
            ],
            "documentation": "https://carta.readthedocs.io/en/latest/installation.html"
        }

    def create_quick_look(self, image_path: str) -> Dict[str, Any]:
        """
        Create quick-look products without CARTA
        Using matplotlib/astropy as fallback

        Args:
            image_path: Path to image

        Returns:
            Quick-look result
        """
        try:
            from astropy.io import fits
            from astropy.wcs import WCS
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm, PowerNorm

            # Read FITS
            hdul = fits.open(image_path)
            data = hdul[0].data
            header = hdul[0].header

            # Handle multi-dimensional data
            if data.ndim > 2:
                data = data[0, 0] if data.ndim == 4 else data[0]

            # Create WCS
            wcs = WCS(header, naxis=2)

            # Create figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection=wcs)

            # Determine scaling
            vmin = np.percentile(data[~np.isnan(data)], 1)
            vmax = np.percentile(data[~np.isnan(data)], 99)

            # Plot
            im = ax.imshow(data, origin='lower', cmap='viridis',
                         norm=LogNorm(vmin=max(vmin, 1e-6), vmax=vmax))

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(header.get('BUNIT', 'Flux'), rotation=270, labelpad=15)

            # Add grid
            ax.coords.grid(True, color='white', ls='--', alpha=0.3)
            ax.coords[0].set_axislabel('Right Ascension')
            ax.coords[1].set_axislabel('Declination')

            # Add beam if present
            if 'BMAJ' in header and 'BMIN' in header:
                from matplotlib.patches import Ellipse
                beam = Ellipse((0.1, 0.1),
                             header['BMAJ'], header['BMIN'],
                             angle=header.get('BPA', 0),
                             transform=ax.transAxes,
                             facecolor='white', edgecolor='black')
                ax.add_patch(beam)

            # Save
            output_path = str(Path(image_path).with_suffix('.quicklook.png'))
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            hdul.close()

            return {
                "status": "success",
                "quicklook": output_path,
                "message": f"Quick-look image saved to {output_path}"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to create quick-look: {str(e)}"
            }


class CARTARemote:
    """Interface for remote CARTA servers"""

    def __init__(self, server_url: str, token: Optional[str] = None):
        """
        Initialize remote CARTA connection

        Args:
            server_url: URL of CARTA server
            token: Authentication token if required
        """
        self.server_url = server_url
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers['Authorization'] = f'Bearer {token}'

    def get_server_info(self) -> Dict[str, Any]:
        """Get information about remote CARTA server"""
        try:
            response = self.session.get(f"{self.server_url}/api/info")
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "code": response.status_code,
                    "message": "Failed to get server info"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def list_available_files(self) -> List[str]:
        """List files available on remote server"""
        try:
            response = self.session.get(f"{self.server_url}/api/files")
            if response.status_code == 200:
                return response.json().get('files', [])
            else:
                return []
        except:
            return []


class MockCARTAVisualizer:
    """Mock CARTA visualizer for when CARTA is not installed"""

    @staticmethod
    def create_demo_visualization(image_type: str = "continuum") -> Dict[str, Any]:
        """Create demo visualization configuration"""

        demos = {
            "continuum": {
                "description": "Radio continuum image",
                "typical_features": [
                    "Point sources (stars, AGN)",
                    "Extended emission (galaxies, nebulae)",
                    "Diffuse background"
                ],
                "recommended_scaling": "logarithmic",
                "recommended_colormap": "viridis",
                "typical_range": "3-sigma to 99.5 percentile"
            },
            "spectral_line": {
                "description": "Spectral line cube",
                "typical_features": [
                    "Emission/absorption lines",
                    "Velocity structure",
                    "Line profiles"
                ],
                "recommended_scaling": "linear",
                "recommended_colormap": "rainbow",
                "typical_range": "-3 to +10 sigma"
            },
            "polarization": {
                "description": "Polarization maps",
                "typical_features": [
                    "Stokes I (total intensity)",
                    "Stokes Q, U (linear polarization)",
                    "Polarization vectors",
                    "Faraday rotation"
                ],
                "recommended_scaling": "linear",
                "recommended_colormap": "seismic",
                "typical_range": "Full range for Q,U"
            }
        }

        return demos.get(image_type, demos["continuum"])