"""
Teacher Agent Module
vLLM + JSON/Neo4j knowledge-enhanced Teacher system
Used to compute KG-enhanced Logits and reward signals

Optimization: Prioritizes JSON dictionary lookup, Neo4j as fallback
"""

import os
import sys
import json
import re
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import requests
from openai import OpenAI

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VLLM_CONFIG, NEO4J_CONFIG, DistillConfig, TEACHER_GT_CONFIG


class GTCache:
    """Ground Truth cache (prioritizes JSON dictionary lookup)"""
    
    def __init__(
        self, 
        json_path: str = None,
        use_json_cache: bool = True,
        use_neo4j_fallback: bool = False,
        neo4j_config: Dict = None,
    ):
        self.use_json_cache = use_json_cache
        self.use_neo4j_fallback = use_neo4j_fallback
        
        # JSON dictionary (primary method)
        self.gt_dict = {}
        if use_json_cache and json_path:
            self._load_json_cache(json_path)
        
        # Neo4j connection (backup method)
        self.neo4j_driver = None
        self.neo4j_config = neo4j_config or NEO4J_CONFIG
        if use_neo4j_fallback:
            self._init_neo4j()
    
    def _load_json_cache(self, json_path: str):
        """Load JSON data to memory dictionary"""
        print(f"Loading JSON cache: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Build index: {(ecuid, dtc): output}
            for item in data:
                input_str = item.get('input', '')
                output = item.get('output', {})
                
                ecuid, dtc = self._parse_input(input_str)
                if ecuid and dtc:
                    self.gt_dict[(ecuid, dtc)] = output
            
            print(f"JSON cache loaded: {len(self.gt_dict)} records")
        except Exception as e:
            print(f"Failed to load JSON cache: {e}")
    
    def _parse_input(self, input_str: str) -> Tuple[str, str]:
        """Extract ECUID and DTC from input string"""
        ecuid = ""
        dtc = ""
        
        ecuid_match = re.search(r'ECUID:\s*(\w+)', input_str)
        dtc_match = re.search(r'DTC:\s*(\w+)', input_str)
        
        if ecuid_match:
            ecuid = ecuid_match.group(1)
        if dtc_match:
            dtc = dtc_match.group(1)
        
        return ecuid, dtc
    
    def _init_neo4j(self):
        """Initialize Neo4j connection (backup)"""
        try:
            from neo4j import GraphDatabase
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_config["uri"],
                auth=(self.neo4j_config["username"], self.neo4j_config["password"])
            )
            print("Neo4j backup connection initialized")
        except Exception as e:
            print(f"Neo4j connection initialization failed: {e}")
    
    def query_by_ecuid_dtc(self, ecuid: str, dtc: str) -> Optional[Dict]:
        """
        Query GT by ECUID and DTC.
        Prioritizes JSON cache, falls back to Neo4j if enabled.
        """
        # 1. Prioritize JSON cache
        if self.use_json_cache:
            result = self.gt_dict.get((ecuid, dtc), None)
            if result:
                return {
                    "ecuid": ecuid,
                    "dtc_code": dtc,
                    "fault_description": result.get("FaultDescription", ""),
                    "service_measures": result.get("ServiceMeasures", ""),
                }
        
        # 2. Fallback to Neo4j (if enabled)
        if self.use_neo4j_fallback and self.neo4j_driver:
            try:
                cypher = """
                MATCH (e:ECU {ecuid: $ecuid})-[r:HAS_FAULT]->(d:DTC {code: $dtc})
                RETURN e.ecuid AS ecuid,
                       d.code AS dtc_code,
                       d.description AS dtc_description,
                       r.fault_description AS fault_description,
                       r.trigger AS trigger,
                       r.possible_causes AS possible_causes,
                       r.service_measures AS service_measures,
                       r.fault_effect AS fault_effect
                LIMIT 1
                """
                with self.neo4j_driver.session(database=self.neo4j_config["database"]) as session:
                    result = session.run(cypher, {"ecuid": ecuid, "dtc": dtc})
                    data = result.data()
                    return data[0] if data else None
            except Exception as e:
                print(f"Neo4j query failed: {e}")
        
        return None
    
    def close(self):
        """Close connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Neo4jConnector:
    """Neo4j database connector (kept for compatibility, replaced by GTCache)"""
    
    def __init__(self, config: Dict = None):
        from neo4j import GraphDatabase
        
        config = config or NEO4J_CONFIG
        self.uri = config["uri"]
        self.username = config["username"]
        self.password = config["password"]
        self.database = config["database"]
        
        self.driver = GraphDatabase.driver(
            self.uri, 
            auth=(self.username, self.password)
        )
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def query_by_ecuid_dtc(self, ecuid: str, dtc: str) -> Optional[Dict]:
        """Query precise info by ECUID and DTC"""
        cypher = """
        MATCH (e:ECU {ecuid: $ecuid})-[r:HAS_FAULT]->(d:DTC {code: $dtc})
        RETURN e.ecuid AS ecuid,
               d.code AS dtc_code,
               d.description AS dtc_description,
               r.fault_description AS fault_description,
               r.trigger AS trigger,
               r.possible_causes AS possible_causes,
               r.service_measures AS service_measures,
               r.fault_effect AS fault_effect
        LIMIT 1
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, {"ecuid": ecuid, "dtc": dtc})
            data = result.data()
            return data[0] if data else None
    
    def validate_response(self, ecuid: str, dtc: str, response: Dict) -> Tuple[bool, float]:
        """Validate student response against KG data"""
        kg_data = self.query_by_ecuid_dtc(ecuid, dtc)
        
        if not kg_data:
            return True, 0.0
        
        student_fault = response.get("FaultDescription", "")
        student_measures = response.get("ServiceMeasures", "")
        
        kg_fault = kg_data.get("fault_description", "")
        kg_measures = kg_data.get("service_measures", "")
        
        fault_match = self._fuzzy_match(student_fault, kg_fault)
        measures_match = self._fuzzy_match(student_measures, kg_measures)
        
        score = (fault_match + measures_match) / 2.0
        is_correct = score > 0.5
        
        return is_correct, score
    
    def _fuzzy_match(self, text1: str, text2: str) -> float:
        """Fuzzy match, returns 0-1 similarity score"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TeacherAgent:
    """
    Teacher Agent: Uses vLLM + JSON Cache/Neo4j for knowledge-enhanced guidance.
    Optimization: Prioritizes JSON dictionary lookup for faster queries.
    """
    
    def __init__(
        self,
        vllm_base_url: str = None,
        model_name: str = None,
        use_json_cache: bool = None,
        use_neo4j_fallback: bool = None,
        json_cache_path: str = None,
        neo4j_config: Dict = None,
    ):
        # vLLM client
        vllm_base_url = vllm_base_url or f"http://{VLLM_CONFIG['host']}:{VLLM_CONFIG['port']}/v1"
        self.client = OpenAI(
            base_url=vllm_base_url,
            api_key="EMPTY",
        )
        self.model_name = model_name or VLLM_CONFIG["model"]
        
        # GT cache (prioritize JSON, Neo4j as backup)
        use_json_cache = use_json_cache if use_json_cache is not None else TEACHER_GT_CONFIG["use_json_cache"]
        use_neo4j_fallback = use_neo4j_fallback if use_neo4j_fallback is not None else TEACHER_GT_CONFIG["use_neo4j_fallback"]
        json_cache_path = json_cache_path or TEACHER_GT_CONFIG["json_cache_path"]
        
        self.gt_cache = GTCache(
            json_path=json_cache_path,
            use_json_cache=use_json_cache,
            use_neo4j_fallback=use_neo4j_fallback,
            neo4j_config=neo4j_config or NEO4J_CONFIG,
        )
        
        # Load ECU knowledge base
        self._load_ecu_knowledge()
        
        # System prompt
        self.system_prompt = """You are a professional automotive fault diagnosis expert. Based on the given diagnostic information and knowledge graph facts, provide accurate fault diagnosis results.

Please output strictly in the following JSON format:
{
  "FaultDescription": "fault description",
  "ServiceMeasures": "repair recommendations"
}

Important: You must answer based on the provided knowledge graph facts, do not fabricate information."""
    
    def _load_ecu_knowledge(self):
        """Load ECU knowledge base"""
        ecuid_name_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "ecuid_name.json"
        )
        
        self.ecuid_to_name = {}
        if os.path.exists(ecuid_name_path):
            try:
                with open(ecuid_name_path, "r", encoding="utf-8") as f:
                    ecuid_data = json.load(f)
                
                for item in ecuid_data:
                    ecuid = item.get("ECUID", "").upper()
                    ecuname = item.get("ECUNAME", "")
                    if ecuid and ecuname:
                        self.ecuid_to_name[ecuid] = ecuname
                
                print(f"[TeacherAgent] ECU knowledge base loaded: {len(self.ecuid_to_name)} ECUIDs")
            except Exception as e:
                print(f"[TeacherAgent] Failed to load ECU knowledge base: {e}")
        else:
            print(f"[TeacherAgent] ECU knowledge base file not found: {ecuid_name_path}")
        
        # Import ECU validation function
        try:
            from data.ecu_knowledge import validate_ecu_in_response
            self.validate_ecu_in_response = validate_ecu_in_response
        except ImportError:
            print("[TeacherAgent] Cannot import ecu_knowledge module, ECU validation disabled")
            self.validate_ecu_in_response = None
    
    def get_kg_context(self, ecuid: str, dtc: str) -> str:
        """Get context info from KG/JSON (prioritizes JSON cache)"""
        kg_data = self.gt_cache.query_by_ecuid_dtc(ecuid, dtc)
        
        if not kg_data:
            return "No relevant information found in knowledge graph."
        
        context = f"""[Knowledge Graph Facts]
ECUID: {kg_data.get('ecuid', 'N/A')}
DTC: {kg_data.get('dtc_code', 'N/A')}
Fault Description: {kg_data.get('fault_description', 'N/A')}
Service Measures: {kg_data.get('service_measures', 'N/A')}"""
        
        return context
    
    def generate_with_kg(
        self,
        ecuid: str,
        dtc: str,
        trigger: str = "",
        time_condition: str = "",
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> Tuple[str, Dict]:
        """
        Generate response with KG enhancement.
        Returns: (generated response, KG data)
        """
        kg_context = self.get_kg_context(ecuid, dtc)
        kg_data = self.gt_cache.query_by_ecuid_dtc(ecuid, dtc)
        
        user_message = f"""Please diagnose the following fault:
ECUID: {ecuid}
DTC: {dtc}
Trigger Condition: {trigger}
Time Condition: {time_condition}

{kg_context}

Please provide the diagnosis result in JSON format based on the above knowledge graph facts."""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content, kg_data
    
    def compute_reward(
        self,
        student_response: str,
        ecuid: str,
        dtc: str,
        config: DistillConfig = None,
    ) -> Tuple[float, Dict]:
        """
        Compute reward for student response.
        Includes: KG match reward + format reward
        Returns: (reward value, detailed info)
        """
        config = config or DistillConfig()
        reward = 0.0
        info = {
            "json_valid": False,
            "kg_match": False,
            "kg_score": 0.0,
        }
        
        # 1. Check JSON format
        try:
            parsed = json.loads(student_response)
            if isinstance(parsed, dict):
                info["json_valid"] = True
                reward += 1.0
            else:
                parsed = {}
        except json.JSONDecodeError:
            try:
                start = student_response.find("{")
                end = student_response.rfind("}") + 1
                if start >= 0 and end > start:
                    extracted = json.loads(student_response[start:end])
                    if isinstance(extracted, dict):
                        parsed = extracted
                        info["json_valid"] = True
                        reward += 0.5
                    else:
                        parsed = {}
                else:
                    parsed = {}
            except:
                parsed = {}
        
        # 2. KG match validation (using gt_cache)
        if config.use_kg_reward and info["json_valid"]:
            is_match, score = self._validate_response(ecuid, dtc, parsed)
            info["kg_match"] = is_match
            info["kg_score"] = score
            
            if is_match:
                reward += config.kg_reward_weight * score
            else:
                reward += config.kg_penalty_weight * (1 - score)
        
        return reward, info
    
    def _validate_response(self, ecuid: str, dtc: str, response: Dict) -> Tuple[bool, float]:
        """
        Validate student response against cached GT data.
        Includes: text match + ECU validation
        Returns: (is_match, match_score)
        """
        kg_data = self.gt_cache.query_by_ecuid_dtc(ecuid, dtc)
        
        student_fault = response.get("FaultDescription", "")
        student_measures = response.get("ServiceMeasures", "")
        full_response = student_fault + " " + student_measures
        
        # 1. Text match (compare with GT)
        text_match_score = 0.5
        if kg_data:
            kg_fault = kg_data.get("fault_description", "")
            kg_measures = kg_data.get("service_measures", "")
            
            fault_match = self._fuzzy_match(student_fault, kg_fault)
            measures_match = self._fuzzy_match(student_measures, kg_measures)
            text_match_score = (fault_match + measures_match) / 2.0
        
        # 2. ECU validation
        ecu_match_score = 0.5
        if self.validate_ecu_in_response is not None:
            ecuid_upper = ecuid.upper() if ecuid else ""
            correct_ecu_name = self.ecuid_to_name.get(ecuid_upper, "")
            
            if correct_ecu_name:
                ecu_match_score = self.validate_ecu_in_response(full_response, correct_ecu_name)
        
        # 3. Combined score (ECU validation weighted higher)
        score = 0.3 * text_match_score + 0.7 * ecu_match_score
        is_correct = score > 0.4
        
        return is_correct, score
    
    def compute_ecu_reward(
        self,
        student_response: str,
        ecuid: str,
    ) -> float:
        """
        Compute ECU validation reward separately.
        Returns: ECU match score (0.0 - 1.0)
        """
        if self.validate_ecu_in_response is None:
            return 0.5
        
        ecuid_upper = ecuid.upper() if ecuid else ""
        correct_ecu_name = self.ecuid_to_name.get(ecuid_upper, "")
        
        if not correct_ecu_name:
            return 0.5
        
        return self.validate_ecu_in_response(student_response, correct_ecu_name)
    
    def _fuzzy_match(self, text1: str, text2: str) -> float:
        """Improved fuzzy match using string similarity"""
        if not text1 or not text2:
            return 0.0
        
        from difflib import SequenceMatcher
        
        # Method 1: Overall string similarity
        overall_sim = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # Method 2: Word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if words1 and words2:
            word_sim = len(words1 & words2) / len(words1 | words2)
        else:
            word_sim = 0.0
        
        return max(overall_sim, word_sim)
    
    def close(self):
        """Close connections"""
        self.gt_cache.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== Test ====================

def test_teacher_agent():
    """Test Teacher Agent"""
    print("=" * 60)
    print("Testing Teacher Agent")
    print("=" * 60)
    
    agent = TeacherAgent()
    
    test_cases = [
        {"ecuid": "0087", "dtc": "5D0C", "trigger": "A fault in the control unit was detected.", "time_condition": "5 s"},
    ]
    
    for case in test_cases:
        print(f"\nTest case: ECUID={case['ecuid']}, DTC={case['dtc']}")
        print("-" * 40)
        
        kg_context = agent.get_kg_context(case["ecuid"], case["dtc"])
        print(f"KG Context:\n{kg_context}")
        
        try:
            response, kg_data = agent.generate_with_kg(
                ecuid=case["ecuid"],
                dtc=case["dtc"],
                trigger=case["trigger"],
                time_condition=case["time_condition"],
            )
            print(f"\nTeacher Response:\n{response}")
            
            reward, info = agent.compute_reward(
                response, case["ecuid"], case["dtc"]
            )
            print(f"\nReward calculation: reward={reward:.2f}, info={info}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    agent.close()
    print("\nTest complete!")


if __name__ == "__main__":
    test_teacher_agent()
