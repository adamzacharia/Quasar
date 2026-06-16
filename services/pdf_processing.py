"""
PDF Processing Service — Handles downloading and extracting text from PDFs.

CALLED BY: core/agent.py (tool: reproduce_paper_methods)
"""

import os
import requests
import tempfile
import logging
from typing import Optional, Dict, Any
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

class PDFProcessingService:
    """Service for downloading and processing PDF documents."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("TACC_API_KEY")
        self.model = os.getenv("QUASAR_PDF_MODEL") or os.getenv("QUASAR_FAST_MODEL", "gpt-4o-mini")
        self.client = LLMClient(model=self.model)
        
    def download_pdf(self, url: str) -> Optional[str]:
        """Download a PDF from a URL to a temporary file and return the path."""
        try:
            logger.info("Downloading PDF from %s", url)
            headers = {"User-Agent": "Mozilla/5.0"} # To avoid blocks from arxiv
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Create a temporary file that isn't automatically deleted upon closing
            fd, temp_path = tempfile.mkstemp(suffix=".pdf")
            with os.fdopen(fd, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return temp_path
            
        except Exception as e:
            logger.error("Failed to download PDF: %s", e)
            return None
            
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract all text from a local PDF file."""
        try:
            loader = PyMuPDF4LLMLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        except Exception as e:
            logger.error("Failed to extract text from PDF: %s", e)
            return ""
            
    def _truncate_text(self, text: str, max_chars: int = 400000) -> str:
        """Truncate text to fit within model context windows (~128k tokens for gpt-4o)."""
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[Text Truncated due to length]..."
        return text

    def extract_methodology(self, full_text: str) -> Dict[str, Any]:
        """
        Use an LLM to heuristically extract ONLY the methodology / data reduction 
        sections of the paper to save context space.
        """
        if not self.client:
            return {"success": False, "error": "OpenAI API key not configured."}
            
        if not full_text.strip():
            return {"success": False, "error": "Provided text is empty."}
            
        truncated_text = self._truncate_text(full_text)
        
        system_prompt = (
            "You are an expert at extracting specific sections from scientific astrophysics papers. "
            "Your task is to read the provided paper text and extract ONLY the sections describing "
            "the 'Observations', 'Data Reduction', or 'Methodology'. Focus on paragraph text containing "
            "instrument configurations, calibration steps, imaging parameters (e.g., CASA tclean settings, "
            "weighting, robust values, tapering, RMS noise levels)."
            "Do NOT include the Introduction, Results, Discussion, or Conclusion. "
            "Return only the relevant extracted text. If the methodology cannot be found, return just 'METHODOLOGY_NOT_FOUND'."
        )
        
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=f"Extract the methodology from this paper:\n\n{truncated_text}",
                temperature=0.0,
                max_output_tokens=4000
            )
            
            extracted_text = response.output_text.strip()
            
            if extracted_text == "METHODOLOGY_NOT_FOUND":
                return {"success": False, "error": "Could not identify a clear methodology section in the text."}
                
            return {
                "success": True,
                "methodology": extracted_text
            }
            
        except Exception as e:
            logger.error("LLM extraction failed: %s", e)
            return {"success": False, "error": str(e)}

    def get_paper_methodology_from_url(self, url: str) -> Dict[str, Any]:
        """End-to-end pipeline: Download -> Extract Full Text -> Extract Methodology."""
        pdf_path = self.download_pdf(url)
        if not pdf_path:
            return {"success": False, "error": "Failed to download PDF from URL."}
            
        try:
            full_text = self.extract_text_from_pdf(pdf_path)
            if not full_text:
                return {"success": False, "error": "Failed to extract text from the downloaded PDF."}
                
            result = self.extract_methodology(full_text)
            return result
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
            except Exception as e:
                logger.warning("Failed to clean up temp file %s: %s", pdf_path, e)

    def query_paper_pdf(self, url: str, query: str) -> Dict[str, Any]:
        """Download a PDF, extract its text, and use an LLM to answer a specific query."""
        if not self.client:
            return {"success": False, "error": "OpenAI API key not configured."}

        pdf_path = self.download_pdf(url)
        if not pdf_path:
            return {"success": False, "error": "Failed to download PDF from URL."}

        try:
            full_text = self.extract_text_from_pdf(pdf_path)
            if not full_text:
                return {"success": False, "error": "Failed to extract text from the downloaded PDF."}

            truncated_text = self._truncate_text(full_text)
            
            system_prompt = (
                "You are an expert astrophysics research assistant. "
                "Your task is to concisely answer the user's question based strictly on the text "
                "extracted from the provided scientific paper. If the answer cannot be found in the text, "
                "state that clearly rather than hallucinating."
            )
            
            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=f"Paper text:\n\n{truncated_text}\n\nQuestion: {query}",
                temperature=0.0,
                max_output_tokens=2000
            )

            answer = response.output_text.strip()
            return {"success": True, "answer": answer}

        except Exception as e:
            logger.error("LLM QA failed: %s", e)
            return {"success": False, "error": str(e)}
        finally:
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
            except Exception as e:
                logger.warning("Failed to clean up temp file %s: %s", pdf_path, e)

