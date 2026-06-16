# services/proposal_critic.py
import os
from typing import Dict, Any, Optional, Callable
from core.llm_client import LLMClient
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from core.prompts import (
    RED_TEAM_TAC_PROMPT,
    FACT_CHECKER_PROMPT,
    RUBRIC_GRADER_PROMPT,
    SYNTHESIZER_PROMPT
)
from services.rag_service import RAGService
from integrations.ads_client import ADSService
import asyncio

class ProposalCriticService:
    """
    Rigorously reviews user-submitted observing proposals using 'Red Team' critique.
    Extracts text from PDF drafts and evaluates them against established observatory rubrics.
    """
    
    def __init__(self, api_key: Optional[str] = None, ads_service: Optional[ADSService] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("TACC_API_KEY")
        self.model = os.getenv("QUASAR_PROPOSAL_CRITIC_MODEL") or os.getenv("QUASAR_REASONING_MODEL", "gpt-4o")
        self.client = LLMClient(model=self.model)
        self.ads_client = ads_service or ADSService()

    def _run_fact_checker(self, proposal_text: str) -> str:
        """
        Agent 1: The Fact Checker.
        Searches NASA ADS based on the proposal text to verify citations and scientific claims.
        """
        # Step 1: Extract key topics/claims to search for. For simplicity, we just use a focused query based on the text.
        # In a more complex implementation, we could have an LLM extract specific queries.
        # Here we'll just extract a very brief summary query to search ADS with, or just search for the title/main topic.
        
        # To keep it robust, let's ask the LLM to generate 1 or 2 key ADS search queries from the proposal
        extraction_prompt = "Generate a single relevant NASA ADS search query (max 20 chars) to fact-check this proposal. Focus on the main astronomical object or method. Return ONLY the search query string, nothing else."
        
        try:
            extraction_res = self.client.responses.create(
                model=self.model,
                instructions="You are a helpful assistant. Output only the requested string.",
                input=f"{extraction_prompt}\n\nProposal Snippet:\n{proposal_text[:2000]}",
                temperature=0.1,
                max_output_tokens=30
            )
            search_query = extraction_res.output_text.strip().strip("'\"")
            
            # Step 2: Search ADS
            literature_context = "No literature found."
            if search_query:
                try:
                    ads_results = self.ads_client.search_papers(query=search_query, max_results=3)
                    if not ads_results.get("error"):
                         papers = ads_results.get("papers", [])
                         if papers:
                             literature_context = "Recent relevant literature retrieved from NASA ADS:\n\n"
                             for p in papers:
                                 literature_context += f"- {p.get('title', '')} by {p.get('author_str', '')} ({p.get('year', '')}): {p.get('abstract', '')}\n"
                except Exception as e:
                    literature_context = f"Error retrieving literature: {e}"
        except Exception:
            literature_context = "Could not generate fact-check queries."
            
        # Step 3: Run the Fact Checker Agent
        prompt = FACT_CHECKER_PROMPT.format(
            proposal_text=proposal_text,
            literature_context=literature_context
        )
        
        response = self.client.responses.create(
            model=self.model,
            instructions="You are the rigorous Fact Checker agent.",
            input=prompt,
            temperature=0.2, # Low temperature for factual analysis
            max_output_tokens=2000
        )
        
        return response.output_text

    def _run_rubric_grader(self, proposal_text: str, rag_service: RAGService) -> str:
        """
        Agent 2: The Rubric Grader.
        Evaluates the proposal against detailed observing rubrics.
        """
        rubric_docs = rag_service.search_rubrics(
            query="ALMA proposal review criteria assessing technical feasibility scientific justification weaknesses",
            k=4
        )
        
        # If no rubrics found, fall back
        if rubric_docs:
            rubric_context = "\n".join([f"- From {doc.metadata.get('source_file', 'Rubric')}:\n{doc.page_content}" for doc in rubric_docs])
        else:
            rubric_context = "No specific rubrics loaded. Rely on general ALMA and NRAO Time Allocation Committee best practices: Check RMS sensitivity vs time, target visibility, and clarity of scientific goals."
            
        prompt = RUBRIC_GRADER_PROMPT.format(
            rubric_context=rubric_context,
            proposal_text=proposal_text
        )
        
        response = self.client.responses.create(
            model=self.model,
            instructions="You are the strict Rubric Grader agent.",
            input=prompt,
            temperature=0.2,
            max_output_tokens=2000
        )
        
        return response.output_text

    def _run_synthesizer(self, fact_check_report: str, rubric_report: str) -> str:
        """
        Agent 3: The Synthesizer.
        Combines reports into final critique.
        """
        prompt = SYNTHESIZER_PROMPT.format(
            fact_check_report=fact_check_report,
            rubric_report=rubric_report
        )
        
        response = self.client.responses.create(
            model=self.model,
            instructions="You are the Synthesizer and Lead Chair of the TAC.",
            input=prompt,
            temperature=0.4,
            max_output_tokens=2500
        )
        
        return response.output_text

    def review_proposal(
        self, 
        file_path: str, 
        rag_service: RAGService, 
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Main pipeline: Extract text from a proposal PDF, then orchestrate the multi-agent committee.
        """
        try:
            filename = os.path.basename(file_path)
            
            # 1. Load and extract text from the PDF
            if progress_callback: progress_callback(f"Loading {filename}...", 10)
            
            loader = PyMuPDF4LLMLoader(file_path)
            documents = loader.load()
            
            if not documents:
                return {"success": False, "error": "No text could be extracted from the PDF."}
                
            proposal_text = "\n".join([doc.page_content for doc in documents])
            
            # Safety check: GPT-4o has a 128k token context window. We'll truncate characters to ~300k.
            max_chars = 300000
            if len(proposal_text) > max_chars:
                proposal_text = proposal_text[:max_chars] + "\n...[Text Truncated due to length]..."
            
            # 2. Run Sub-Agents
            # For simplicity in this synchronous method, we run them sequentially. 
            # In an async method, we could use asyncio.gather for the first two.
            if progress_callback: progress_callback("Fact Checker: Verifying citations and claims...", 30)
            fact_check_report = self._run_fact_checker(proposal_text)
            
            if progress_callback: progress_callback("Rubric Grader: Evaluating against observatory rubrics...", 60)
            rubric_report = self._run_rubric_grader(proposal_text, rag_service)
            
            if progress_callback: progress_callback("Synthesizer: Drafting final Red Team critique...", 85)
            critique = self._run_synthesizer(fact_check_report, rubric_report)
            
            if progress_callback: progress_callback(f"✓ Committee review completed for {filename}", 100)
            
            return {
                "success": True,
                "filename": filename,
                "critique": critique,
                "fact_check_report": fact_check_report,
                "rubric_report": rubric_report,
                "pages_analyzed": len(documents)
            }

        except Exception as e:
            if progress_callback: progress_callback(f"✗ Failed: {str(e)}", 0)
            return {"success": False, "error": str(e)}
