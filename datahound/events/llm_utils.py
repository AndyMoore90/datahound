"""LLM integration utilities for event analysis using DeepSeek"""

import json
import os
import re
import time
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class LLMConfig:
    """Configuration for LLM analysis using DeepSeek"""
    api_key: Optional[str] = None
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    max_tokens: int = 8192
    temperature: float = 0.0
    timeout: int = 30
    max_retries: int = 2
    max_concurrent_requests: int = 10


class DeepSeekClient:
    """DeepSeek client adapted from legacy code"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self.model = config.model
        self.max_retries = config.max_retries
        self._setup_client()
    
    def _setup_client(self):
        """Setup DeepSeek client via OpenAI interface"""
        try:
            import openai
            
            # Try multiple sources for API key
            api_key = (
                self.config.api_key or 
                self._load_api_key_from_env_file() or
                self._load_api_key_from_config() or
                os.getenv("DEEPSEEK_API_KEY") or 
                os.getenv("OPENAI_API_KEY")
            )
            
            if api_key:
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url=self.config.base_url
                )
            else:
                self.client = None
                print("Warning: No DeepSeek API key found. See Events page for setup options. LLM analysis will be disabled.")
        except ImportError:
            print("Warning: openai package not installed. LLM analysis will be disabled.")
            self.client = None
    
    def _load_api_key_from_env_file(self) -> Optional[str]:
        """Load API key from .env file"""
        try:
            from pathlib import Path
            env_file = Path(".env")
            if env_file.exists():
                content = env_file.read_text(encoding="utf-8")
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("DEEPSEEK_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
        return None
    
    def _load_api_key_from_config(self) -> Optional[str]:
        """Load API key from global config file"""
        try:
            from pathlib import Path
            import json
            
            config_file = Path("config/global.json")
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                return config.get("deepseek_api_key")
        except Exception:
            pass
        return None


class LLMAnalyzer:
    """Base class for LLM-based analysis using DeepSeek"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = DeepSeekClient(config)
    
    def is_available(self) -> bool:
        """Check if LLM analysis is available"""
        return self.client.client is not None
    
    def extract_json_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response using improved logic"""
        if not content:
            return None
        
        # Strategy 1: Look for markdown JSON block (improved regex)
        markdown_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if markdown_match:
            try:
                json_content = markdown_match.group(1).strip()
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Look for any complete JSON object (improved)
        # Find the first { and match to its closing }
        start_idx = content.find('{')
        if start_idx != -1:
            brace_count = 0
            for i, char in enumerate(content[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            json_content = content[start_idx:i+1]
                            return json.loads(json_content)
                        except json.JSONDecodeError:
                            break
        
        # Strategy 3: Look for JSON pattern with required fields (for system age)
        json_match = re.search(r'\{.*?"age".*?"job_date".*?"text_snippet".*?\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Try parsing the entire content as JSON
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass
        
        return None
    
    def analyze_text_with_retries(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Analyze text using LLM with retries, adapted from legacy code"""
        if not self.is_available():
            return None
        
        for attempt in range(self.client.max_retries + 1):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = self.client.client.chat.completions.create(
                    model=self.client.model,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                
                content = response.choices[0].message.content.strip()
                
                # Use robust JSON extraction
                result = self.extract_json_from_response(content)
                if result:
                    return result
                else:
                    # If JSON extraction fails, return raw content
                    return {"raw_response": content}
                    
            except Exception as e:
                print(f"LLM API attempt {attempt + 1} failed: {e}")
                if attempt < self.client.max_retries:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"All {self.client.max_retries + 1} attempts failed")
                    return None
        
        return None
    
    def analyze_text(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Simple wrapper for analyze_text_with_retries"""
        return self.analyze_text_with_retries(system_prompt, user_prompt)
    
    def analyze_texts_concurrent(self, requests: List[Dict[str, Any]], 
                                progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
        """Analyze multiple texts concurrently using ThreadPoolExecutor
        
        Args:
            requests: List of dicts with keys: 'system_prompt', 'user_prompt', 'id' (optional)
            progress_callback: Optional callback function called with (completed, total)
            
        Returns:
            List of results in same order as requests. Each result has:
            - 'id': request id if provided
            - 'result': LLM result or None if failed
            - 'success': boolean
            - 'duration_ms': processing time in milliseconds
        """
        if not self.is_available():
            return [{"id": req.get("id"), "result": None, "success": False, "duration_ms": 0, "error": "LLM not available"} for req in requests]
        
        if not requests:
            return []
        
        results = [None] * len(requests)  # Preserve order
        max_workers = min(self.config.max_concurrent_requests, len(requests))
        
        def process_request(req_idx: int, request: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single request"""
            start_time = time.perf_counter()
            try:
                result = self.analyze_text_with_retries(
                    request["system_prompt"], 
                    request["user_prompt"]
                )
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                return {
                    "id": request.get("id"),
                    "result": result,
                    "success": result is not None,
                    "duration_ms": duration_ms,
                    "index": req_idx
                }
            except Exception as e:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                return {
                    "id": request.get("id"),
                    "result": None,
                    "success": False,
                    "duration_ms": duration_ms,
                    "error": str(e),
                    "index": req_idx
                }
        
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            future_to_index = {
                executor.submit(process_request, i, req): i 
                for i, req in enumerate(requests)
            }
            
            # Initial progress callback
            if progress_callback:
                progress_callback(0, len(requests))
            
            # Process completed requests
            for future in as_completed(future_to_index):
                result = future.result()
                results[result["index"]] = result
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(requests))
        
        return results


class SystemAgeAnalyzer(LLMAnalyzer):
    """LLM analyzer for system age from job histories"""
    
    def __init__(self, config: LLMConfig, logger=None):
        super().__init__(config)
        self.logger = logger
    
    def analyze_location_jobs(self, location_id: str, jobs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze job history to determine system age"""
        if not self.is_available():
            return {
                "description": "LLM analysis not available",
                "age": 0,
                "job_date": "",
                "text_snippet": "",
                "reasoning": "OpenAI API not configured"
            }
        
        # Build jobs text from first 10 jobs
        jobs_text = ""
        for i, job in enumerate(jobs[:10]):
            job_summary = job.get('summary', '')
            job_date = job.get('created_date', '')
            if job_summary and job_summary.strip():
                jobs_text += f"Job {i+1} ({job_date}): {job_summary.strip()}\n"
        
        if not jobs_text.strip():
            return {
                "description": "No job summaries found",
                "age": 0,
                "job_date": "",
                "text_snippet": "",
                "reasoning": "No job data available"
            }
        
        system_prompt = (
            "Analyze HVAC job data to find CURRENT system age as of 2025. Follow these rules:\n\n"
            "AGE CALCULATION RULES:\n"
            "1. When you find 'Age of Equipment: X years' in a job summary:\n"
            "   - Add the time elapsed since that job date to get current age\n"
            "   - Example: 'Age of Equipment: 5 years' from 2020 job = 5 + 5 = 10 years current age\n\n"
            "2. When you find 'installed in YEAR' or similar:\n"
            "   - Calculate: 2025 - YEAR = current age\n"
            "   - Example: 'installed in 2015' = 2025 - 2015 = 10 years current age\n\n"
            "3. When system was REPLACED:\n"
            "   - Use replacement date, not original install date\n"
            "   - Calculate from replacement year to 2025\n"
            "   - If text says 'Approximate age of system being replaced: 10+ years old':\n"
            "     * This indicates replacement already happened in that job\n"
            "     * Check if subsequent jobs confirm the replacement was completed\n"
            "     * If replacement completed, count years from replacement date to 2025\n"
            "     * Flag this as 'replacement_detected' for verification\n\n"
            "4. Time calculation guidelines:\n"
            "   - Always compute current age as of 2025\n"
            "   - Round to nearest whole year\n"
            "   - If job is from 2024 and says 'Age: 8 years', current age = 9 years\n\n"
            "5. Always return the CURRENT age as of 2025, not the age mentioned in historical jobs\n\n"
            "RESPONSE FORMAT EXAMPLES:\n"
            "{\"description\": \"Found equipment age reference\", \"age\": 9, \"job_date\": \"2023-05-15\", \"text_snippet\": \"Age of Equipment: 7 years\", \"reasoning\": \"Age from job + years elapsed\", \"confidence\": 0.9, \"source_type\": \"job_history\"}\n"
            "{\"description\": \"System installed in 2018\", \"age\": 7, \"job_date\": \"2022-03-10\", \"text_snippet\": \"Unit was installed in 2018\", \"reasoning\": \"2025 - 2018 = 7 years\", \"confidence\": 0.95, \"source_type\": \"installation_date\"}\n"
            "{\"description\": \"Replacement detected\", \"age\": 3, \"job_date\": \"2022-01-15\", \"text_snippet\": \"Approximate age of system being replaced: 10+ years old\", \"reasoning\": \"Replacement in 2022, current age from replacement date\", \"confidence\": 0.8, \"source_type\": \"replacement_detected\", \"replacement_detected\": true}\n"
            "{\"description\": \"No age information found\", \"age\": 0, \"job_date\": \"\", \"text_snippet\": \"\", \"reasoning\": \"No age indicators in job history\", \"confidence\": 0.0, \"source_type\": \"none\"}\n\n"
            "Return valid JSON only. Do not include explanations outside the JSON."
        )
        
        user_prompt = f"Location ID: {location_id}\n\nJob History:\n{jobs_text}"
        
        # Perform LLM analysis with timing
        llm_start = time.perf_counter()
        result = self.analyze_text(system_prompt, user_prompt)
        llm_duration = int((time.perf_counter() - llm_start) * 1000)
        
        # Log LLM analysis
        if self.logger:
            self.logger.log_llm_analysis(
                location_id, "system_age",
                {"job_count": len(jobs), "text": jobs_text},
                result or {"error": "No result"},
                llm_duration, result is not None
            )
        
        if result is None:
            return {
                "description": "LLM analysis failed",
                "age": 0,
                "job_date": "",
                "text_snippet": "",
                "reasoning": "API call failed"
            }
        
        # Validate and coerce result using legacy logic
        validated_result = self._validate_system_age_result(result)
        return validated_result
    
    def _validate_system_age_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate system age result using legacy validation logic"""
        # Extract and validate fields
        description = str(result.get("description", "")).strip()
        age = result.get("age", 0)
        job_date = str(result.get("job_date", "")).strip()
        text_snippet = str(result.get("text_snippet", "")).strip()
        reasoning = str(result.get("reasoning", "")).strip()
        confidence = float(result.get("confidence", 0.0))
        source_type = str(result.get("source_type", "unknown")).strip()
        replacement_detected = bool(result.get("replacement_detected", False))
        
        # Validate age - clamp to 0-50 range like legacy
        try:
            age = int(float(age))
        except (ValueError, TypeError):
            age = 0
        
        age = max(0, min(50, age))  # Clamp to 0-50 range
        
        # Ensure description is not empty
        if not description:
            description = "No age information found" if age == 0 else "System age detected"
        
        # Clamp confidence to 0.0-1.0 range
        confidence = max(0.0, min(1.0, confidence))
        
        # Return validated result with all fields
        return {
            "description": description,
            "age": age,
            "job_date": job_date,
            "text_snippet": text_snippet,
            "reasoning": reasoning or "No reasoning provided",
            "confidence": confidence,
            "source_type": source_type,
            "replacement_detected": replacement_detected
        }


class PermitReplacementAnalyzer(LLMAnalyzer):
    """LLM analyzer for permit replacement analysis"""
    
    def __init__(self, config: LLMConfig, logger=None):
        super().__init__(config)
        self.logger = logger
    
    def analyze_location_permits(self, location_id: str, permits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze permit history to find system replacements"""
        if not self.is_available():
            return {
                "description": "LLM analysis not available",
                "most_recent_issue_date": "",
                "years_since_replacement": 0,
                "reasoning": "OpenAI API not configured",
                "permits_count": len(permits),
                "most_recent_replacement_contractor": "",
                "has_mccullough_replacement": False,
                "most_recent_replacement_permit_id": ""
            }
        
        if not permits:
            return {
                "description": "No permits found",
                "most_recent_issue_date": "",
                "years_since_replacement": 0,
                "reasoning": "No permit data available",
                "permits_count": 0,
                "most_recent_replacement_contractor": "",
                "has_mccullough_replacement": False,
                "most_recent_replacement_permit_id": ""
            }
        
        # Build permit text
        permits_text = ""
        for i, permit in enumerate(permits[:20]):  # Limit to 20 permits
            permit_id = permit.get('permit_id', f'permit_{i+1}')
            issue_date = permit.get('issue_date', '')
            description = permit.get('description', '')
            contractor = permit.get('contractor', '')
            permits_text += f"Permit {permit_id} ({issue_date}) - {contractor}: {description}\n"
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Use exact legacy prompt
        system_prompt = (
            f"You are analyzing mechanical permit records to detect HVAC system replacement events for a location.\n"
            f"Current date is {current_date}.\n"
            "Task: from the given permit list, identify evidence of a central system replacement (e.g., replace system, full system install).\n"
            "If multiple replacements exist, choose the most recent by issue date.\n"
            "Compute years_since_replacement as an integer using the current date above.\n"
            "If the most recent replacement happened within the last 12 months, set years_since_replacement=1 (never 0).\n"
            "Only use years_since_replacement=0 when no replacement evidence exists and most_recent_issue_date is an empty string.\n"
            "Also return the permit_id for the most recent system replacement.\n"
            "Also identify the contractor/company for the most recent system replacement, if any. If unknown, use an empty string.\n"
            "Also determine whether any permit indicates a system replacement performed by 'McCullough Heating and Air' at any time for this location.\n"
            "Return JSON with fields: description, most_recent_issue_date (MM/DD/YYYY or ISO if available), years_since_replacement (integer), reasoning, permits_count, most_recent_replacement_contractor (string), has_mccullough_replacement (boolean), most_recent_replacement_permit_id (string).\n"
            "If no replacement info is present, return description summarizing permits and set most_recent_issue_date='', years_since_replacement=0, most_recent_replacement_contractor='', has_mccullough_replacement=false, most_recent_replacement_permit_id='', reasoning short."
        )
        
        user_prompt = f"Location ID: {location_id}\n\nPermit History:\n{permits_text}"
        
        result = self.analyze_text(system_prompt, user_prompt)
        if result is None:
            return {
                "description": "LLM analysis failed",
                "most_recent_issue_date": "",
                "years_since_replacement": 0,
                "reasoning": "API call failed",
                "permits_count": len(permits),
                "most_recent_replacement_contractor": "",
                "has_mccullough_replacement": False,
                "most_recent_replacement_permit_id": ""
            }
        
        # Validate and coerce result using legacy logic
        validated_result = self._validate_permit_replacement_result(result, permits)
        return validated_result
    
    def _validate_permit_replacement_result(self, result: Dict[str, Any], permits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate permit replacement result using legacy validation logic"""
        
        # Extract and validate basic fields
        description = str(result.get("description", "")).strip()
        most_recent_issue_date = str(result.get("most_recent_issue_date", "")).strip()
        reasoning = str(result.get("reasoning", "")).strip()
        
        # Validate years_since_replacement
        yrs = result.get("years_since_replacement", 0)
        try:
            yrs = int(float(yrs))
        except (ValueError, TypeError):
            yrs = 0
        
        # Validate contractor name and permit ID
        contractor_name = str(result.get("most_recent_replacement_contractor", "")).strip()
        recent_permit_id = str(result.get("most_recent_replacement_permit_id", "")).strip()
        
        # Legacy logic: If permit_id is provided, try to extract contractor from actual permit data
        if recent_permit_id:
            for permit in permits:
                if str(permit.get('permit_id', '')).strip() == recent_permit_id.strip():
                    extracted_contractor = str(permit.get('Contractor Company Name', '') or permit.get('contractor', ''))
                    if extracted_contractor.strip():
                        contractor_name = extracted_contractor.strip()
                    break
        
        # Validate McCullough flag
        has_mccullough_replacement = bool(result.get("has_mccullough_replacement", False))
        
        # Ensure required fields have defaults
        if not description:
            description = "Permit analysis completed"
        if not reasoning:
            reasoning = "Analysis performed"
        
        return {
            "description": description,
            "most_recent_issue_date": most_recent_issue_date,
            "years_since_replacement": yrs,
            "reasoning": reasoning,
            "permits_count": len(permits),
            "most_recent_replacement_contractor": contractor_name,
            "has_mccullough_replacement": has_mccullough_replacement,
            "most_recent_replacement_permit_id": recent_permit_id
        }


def build_aging_systems_system_prompt(company_name: str, current_date: str) -> str:
    questions = (
        "Questions to answer (keys under answers):\n"
        "1) last_system_install_date: Most recent system install/replacement date at this location.\n"
        "2) go_to_contractor: Contractor most frequently used for installs/replacements (tie-break: most recent).\n"
        "3) used_other_contractor_after_mccullough: Did they use a non-\"" + company_name + "\" contractor AFTER having used \"" + company_name + "\"?\n"
        "4) estimated_current_system_age_years: Estimated current age as of " + current_date + ".\n"
        "5) most_recent_replacement: Details of most recent replacement permit if any.\n"
        "6) mccullough_has_replaced_system_here: Has \"" + company_name + "\" ever performed a replacement here.\n"
        "7) last_permit_issue_date: Most recent permit issue date.\n"
        "8) last_job_date: Most recent job completion/created date.\n"
    )

    schema = (
        "Return exactly one JSON object (no markdown fences):\n"
        "{\n"
        "  \"company\": \"" + company_name + "\",\n"
        "  \"analysis_date\": \"" + current_date + "\",\n"
        "  \"location_id\": string,\n"
        "  \"answers\": {\n"
        "    \"last_system_install_date\": { \"answer\": string|null, \"confidence\": number, \"reasoning_summary\": string, \"evidence_snippets\": [ { \"source\": \"permit|job|aggregate\", \"id\": string, \"date\": string, \"text\": string } ], \"supporting_data\": { \"contractor\": string|null, \"permit_id\": string|null } },\n"
        "    \"go_to_contractor\": { \"answer\": string|null, \"confidence\": number, \"reasoning_summary\": string, \"evidence_snippets\": [ {\"source\": string, \"id\": string, \"date\": string, \"text\": string } ], \"supporting_data\": { \"top_contractors\": [ {\"name\": string, \"count\": number} ], \"window\": \"5y\" } },\n"
        "    \"used_other_contractor_after_mccullough\": { \"answer\": boolean|null, \"confidence\": number, \"reasoning_summary\": string, \"evidence_snippets\": [ {\"source\": string, \"id\": string, \"date\": string, \"text\": string } ], \"supporting_data\": { \"first_other_after_mccullough_date\": string|null, \"last_mccullough_date\": string|null, \"other_contractor\": string|null } },\n"
        "    \"estimated_current_system_age_years\": { \"answer\": number|null, \"confidence\": number, \"reasoning_summary\": string, \"evidence_snippets\": [ {\"source\": string, \"id\": string, \"date\": string, \"text\": string } ], \"supporting_data\": { \"method\": \"age_from_job_plus_elapsed|installed_in_year|replacement_reset|no_signal\", \"reference_date\": string|null } },\n"
        "    \"most_recent_replacement\": { \"answer\": { \"date\": string|null, \"contractor\": string|null, \"permit_id\": string|null }, \"confidence\": number, \"reasoning_summary\": string, \"evidence_snippets\": [ {\"source\": string, \"id\": string, \"date\": string, \"text\": string } ], \"supporting_data\": {} },\n"
        "    \"mccullough_has_replaced_system_here\": { \"answer\": boolean|null, \"confidence\": number, \"reasoning_summary\": string, \"evidence_snippets\": [ {\"source\": string, \"id\": string, \"date\": string, \"text\": string } ], \"supporting_data\": { \"dates\": [string] } },\n"
        "    \"last_permit_issue_date\": { \"answer\": string|null, \"confidence\": number, \"reasoning_summary\": string, \"evidence_snippets\": [ {\"source\": string, \"id\": string, \"date\": string, \"text\": string } ], \"supporting_data\": {} },\n"
        "    \"last_job_date\": { \"answer\": string|null, \"confidence\": number, \"reasoning_summary\": string, \"evidence_snippets\": [ {\"source\": string, \"id\": string, \"date\": string, \"text\": string } ], \"supporting_data\": {} },\n"
        "  }\n"
        "}\n"
    )

    examples = (
        "Examples (one per question):\n"
        "last_system_install_date => { \"answer\": \"2018-07-12\", \"confidence\": 0.85, \"reasoning_summary\": \"Newest permit shows replacement\", \"evidence_snippets\": [ {\"source\": \"permit\", \"id\": \"HV-12345\", \"date\": \"2018-07-12\", \"text\": \"Replace HVAC system\"} ], \"supporting_data\": {\"contractor\": \"" + company_name + "\", \"permit_id\": \"HV-12345\" } }\n"
        "go_to_contractor => { \"answer\": \"" + company_name + "\", \"confidence\": 0.9, \"reasoning_summary\": \"Most permits in last 5y\", \"evidence_snippets\": [ {\"source\": \"permit\", \"id\": \"PR-2\", \"date\": \"2022-04-10\", \"text\": \"Contractor: " + company_name + "\"} ], \"supporting_data\": { \"top_contractors\": [ {\"name\": \"" + company_name + "\", \"count\": 4}, {\"name\": \"OtherCo\", \"count\": 2} ], \"window\": \"5y\" } }\n"
        "used_other_contractor_after_mccullough => { \"answer\": true, \"confidence\": 0.8, \"reasoning_summary\": \"Permit by OtherCo after last " + company_name + " job\", \"evidence_snippets\": [ {\"source\": \"permit\", \"id\": \"PR-9\", \"date\": \"2024-03-01\", \"text\": \"Contractor: OtherCo\"} ], \"supporting_data\": { \"first_other_after_mccullough_date\": \"2024-03-01\", \"last_mccullough_date\": \"2023-06-15\", \"other_contractor\": \"OtherCo\" } }\n"
        "estimated_current_system_age_years => { \"answer\": 12, \"confidence\": 0.85, \"reasoning_summary\": \"Installed in 2013 per job note\", \"evidence_snippets\": [ {\"source\": \"job\", \"id\": \"\", \"date\": \"2019-05-20\", \"text\": \"Installed in 2013\"} ], \"supporting_data\": { \"method\": \"installed_in_year\", \"reference_date\": \"2013-01-01\" } }\n"
        "most_recent_replacement => { \"answer\": { \"date\": \"2022-05-02\", \"contractor\": \"" + company_name + "\", \"permit_id\": \"PR-100\" }, \"confidence\": 0.9, \"reasoning_summary\": \"Latest replacement permit\", \"evidence_snippets\": [ {\"source\": \"permit\", \"id\": \"PR-100\", \"date\": \"2022-05-02\", \"text\": \"Replace HVAC system\"} ], \"supporting_data\": {} }\n"
        "mccullough_has_replaced_system_here => { \"answer\": true, \"confidence\": 0.95, \"reasoning_summary\": \"Replacement permit lists " + company_name + "\", \"evidence_snippets\": [ {\"source\": \"permit\", \"id\": \"PR-100\", \"date\": \"2022-05-02\", \"text\": \"Contractor: " + company_name + "\"} ], \"supporting_data\": { \"dates\": [\"2022-05-02\"] } }\n"
        "last_permit_issue_date => { \"answer\": \"2024-03-01\", \"confidence\": 0.98, \"reasoning_summary\": \"Max issued_date\", \"evidence_snippets\": [ {\"source\": \"permit\", \"id\": \"\", \"date\": \"2024-03-01\", \"text\": \"\"} ], \"supporting_data\": {} }\n"
        "last_job_date => { \"answer\": \"2024-02-18\", \"confidence\": 0.95, \"reasoning_summary\": \"Latest job completion\", \"evidence_snippets\": [ {\"source\": \"job\", \"id\": \"\", \"date\": \"2024-02-18\", \"text\": \"\"} ], \"supporting_data\": {} }\n"
    )

    rules = (
        "Guidelines:\n"
        "- Company name: " + company_name + ". Use this exact name for matching.\n"
        "- Current date: " + current_date + "\n"
        "- Use ONLY the provided YAML content (aggregates, permit_history, job_history). If unknown, set answer=null and confidence=0.0.\n"
        "- Prefer permits for installs/replacements; prefer jobs for maintenance/notes.\n"
        "- Age: if a job note says 'Age of Equipment: X years' on date D, current age = X + elapsed_years(D → " + current_date + "). If 'installed in YEAR', current age = year(" + current_date + ") - YEAR. If replacement after install, age resets at replacement.\n"
        "- Evidence: include up to 4 concise evidence_snippets with relevant text and dates.\n"
        "- Dates: use ISO if available; otherwise use the provided format.\n"
        "- Output: Strict JSON only. No markdown fences, no extra text.\n"
    )

    prompt = (
        "You are Deepseek Reasoner assisting " + company_name + " with HVAC customer insights.\n"
        "You will receive a YAML document for one location containing aggregates, permit_history, and job_history.\n"
        "Analyze it and answer the questions below.\n\n"
        + questions
        + "\n\n" + schema
        + "\n\n" + examples
        + "\n\n" + rules
    )
    return prompt