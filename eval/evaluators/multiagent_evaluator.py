"""
Multi-Agent Diagnostic Evaluator for Automotive Fault Diagnosis

Architecture:
  Step 1: Technical Precision Scoring
  Gate 1: High confidence pass (score >= 92)
  Gate 2: Critical entity matching (ECU/DTC codes)
  Gate 3: Logic conflict detection
  Step 2: Structured extraction + semantic arbitration
"""
import requests
import re
import json
import time
import ast
from typing import Dict, Any, List, Union

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[System] Warning: sentence-transformers not found. Semantic features disabled.")


class ThresholdCalibrationEvaluator:
    def __init__(self, semantic_model_name: str, llm_api_url: str, llm_api_key: str, llm_model_name: str,
                 same_config: Dict[str, str], language: str = "zh"):
        """Initialize evaluator with dual LLM configurations."""
        self.semantic_model_name = semantic_model_name
        self.api_url = llm_api_url
        self.api_key = llm_api_key
        self.model_name = llm_model_name

        # Arbitration LLM for synonym validation (preferably search-enabled)
        self.same_api_url = same_config.get("api_url")
        self.same_api_key = same_config.get("api_key")
        self.same_model_name = same_config.get("model_name")

        self.bert_model = None
        self.language = language

        # BMW ECU knowledge base
        self.BMW_ECU_LIST = {
            # Powertrain & Chassis
            "DME", "DDE", "EGS", "VTG", "GWS", "DSC", "DSCi", "EPS", "AL", "HSR",
            "EHC", "VDP", "EDC", "EARS", "GHAS", "EMF", "PCU",
            # Body & Gateway
            "CAS", "FEM", "BDC", "BDC_BODY", "BDC_GW", "ZGM", "JBE", "REM", "FRM",
            "BCP", "VIP", "FLM", "FLM2", "STML", "STMR", "HKL", "FZD", "IHKA", "IHKA_PRO",
            "SMFA", "SMBF", "SMFAH", "SMBFH",
            # Infotainment & Cockpit
            "HU_H", "HU_NBT", "HU_ENTRY", "HU_CIC", "MGU", "MGU21", "MGU22", "IDC", "IDC23",
            "RAM", "BOOSTER", "KOMBI", "DKOMBI", "DKOMBI4", "DKOMBI8", "ATM", "TCB", "WIB",
            "CON", "ZBE", "TBX",
            # ADAS & Safety
            "ACSM", "MRS", "ICM", "SAS", "SAS2", "SAS3", "KAFAS", "KAFAS2", "KAFAS4",
            "TRSVC", "ICAM", "ADCAM", "PMA", "PAD", "LRR", "SRR", "RSU",
            # EV & High Voltage
            "EME", "SME", "HVS", "CSC", "CCU", "KLE", "LIM", "EWS", "ZKE", "GM"
        }

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"      [Evaluator] Pre-loading Semantic Model: {self.semantic_model_name}...")
            self._load_bert_model()

    def _load_bert_model(self):
        """Load BERT model for semantic similarity computation."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE: return
        if self.bert_model is None:
            try:
                self.bert_model = SentenceTransformer(self.semantic_model_name)
            except Exception as e:
                print(f"      [Evaluator] Error loading BERT: {e}")

    def _call_llm_generic(self, url, key, model, prompt, temp=0.0, max_retries=3) -> str:
        """Generic LLM caller with retry and backoff mechanism."""
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temp}

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=30)

                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    # Clean markdown artifacts
                    if content.startswith("```json"):
                        content = content.replace("```json", "").replace("```", "")
                    elif content.startswith("```"):
                        content = content.replace("```", "")
                    return content.strip()

                elif resp.status_code == 429 or (500 <= resp.status_code < 600):
                    print(f"      [LLM Warning] Rate Limit/Server Error {resp.status_code} (Attempt {attempt + 1}/{max_retries}). Retrying...")
                else:
                    print(f"      [LLM Error] Status: {resp.status_code}, Msg: {resp.text}")
                    return ""

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                print(f"      [Network Error] Timeout/Connection failed (Attempt {attempt + 1}/{max_retries}): {e}")
            except Exception as e:
                print(f"      [LLM Exception] {e}")

            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)
                time.sleep(wait_time)

        print(f"      [System] LLM Call Failed after {max_retries} retries.")
        return ""

    def compute_similarity(self, text1, text2):
        """Compute semantic similarity using BERT embeddings."""
        if not text1 or not text2: return 0.0
        try:
            if self.bert_model is None: self._load_bert_model()
            e1 = self.bert_model.encode(text1, convert_to_tensor=True)
            e2 = self.bert_model.encode(text2, convert_to_tensor=True)
            return float(util.pytorch_cos_sim(e1, e2)[0][0])
        except:
            return 0.0

    def _build_prompt_score(self, field, groundtruth, prediction):
        """Build scoring prompt based on field type."""
        if field == "FaultDescription":
            return f"""Role: You are a Senior BMW Master Technician with 20+ years of experience.
Task: Evaluate the predicted Fault Description against the ISTA Standard with extreme precision.

Input Data:
[groundtruth (Standard)]: "{groundtruth}"
[prediction (Model)]: "{prediction}"

**EVALUATION RUBRIC:**

**TIER 1: FATAL ERRORS (Score: 0)**
- Topology Danger: Wrong Cylinder/Bank/Wheel
- System Hallucination: Wrong system entirely
- Physics Contradiction: Opposite electrical states

**TIER 2: SCORING GRADIENT:**
* 98-100: Semantically identical
* 90-97: Technical equivalence (synonyms valid)
* 80-89: Correct component, vague status
* 60-79: Correct system, vague component
* < 60: Misleading/wrong component

Output: Score: <Int>"""

        else:  # RepairMeasures
            return f"""Role: Senior BMW Master Technician & Warranty Auditor.
Task: Audit technical accuracy of diagnostic prediction against ISTA Standard.

Input Data:
[Field]: {field}
[Standard (groundtruth)]: "{groundtruth}"
[Technician Report (prediction)]: "{prediction}"

**SCORING PROTOCOL (Start at 100, apply deductions):**

**FATAL ERRORS (-100 pts, Score=0):**
- HV Safety Violation
- Topology Mismatch (Bank/Side)
- Mechanism Reversal (Short to Battery vs Ground)
- Unauthorized Upsell (Diagnosis -> Replace without condition)

**CRITICAL ERRORS (-40 pts):**
- Scope Expansion (Diagnosis task but suggests Replace)
- Critical Omission (Missing power/voltage check)
- Action Severity Mismatch (Replace vs Repair)

**MINOR ERRORS (-10 pts each):**
- Terminology inconsistency
- Code typos

**PRECISION DEDUCTIONS (-30 pts):**
- Component generalization
- Vague conditions

**TOLERANCE (No deduction):**
- Pre-step checks before repair (valid SOP)
- Specificity (specific covers general)
- Module equivalency (FEM=BDC)
- Wiring synonyms
- Conditional statements ("if faulty")

Output: Reason: [Brief justification]. Score: [0-100]"""

    def _check_hard_entities(self, groundtruth, prediction) -> bool:
        """Check for critical identifiers: DTCs, versions, ECU names."""
        regex_p_code = r'\bP[0-9][0-9A-Fa-f]{3}\b'
        regex_hex = r'\b0x[0-9A-Fa-f]{3,8}\b'
        regex_bmw_dtc = r'\b[0-9A-Fa-f]{6,8}\b'
        regex_ver = r'\b\d+\.\d+\.\d+\b'
        regex_caps = r'\b[A-Z]{3,}[_A-Z0-9]*\b'

        groundtruth_tokens = set()
        flags = re.IGNORECASE
        groundtruth_tokens.update(re.findall(regex_p_code, groundtruth, flags))
        groundtruth_tokens.update(re.findall(regex_hex, groundtruth, flags))
        groundtruth_tokens.update(re.findall(regex_bmw_dtc, groundtruth, flags))
        groundtruth_tokens.update(re.findall(regex_ver, groundtruth, flags))

        # Filter valid BMW ECU modules
        potential_modules = re.findall(regex_caps, groundtruth)
        valid_modules = {m for m in potential_modules if m in self.BMW_ECU_LIST}
        groundtruth_tokens.update(valid_modules)

        if not groundtruth_tokens:
            return False

        # Check with equivalence mapping
        EQUIV_MAP = {
            "FEM": ["BDC", "BDC_BODY", "CAS", "JBE", "ZGM"],
            "BDC": ["FEM", "BDC_BODY", "CAS", "JBE"],
            "BDC_BODY": ["FEM", "BDC", "CAS"],
            "CAS": ["FEM", "BDC", "BDC_BODY"],
            "DME": ["ECU", "ECM", "MOTOR"],
            "KOMBI": ["DKOMBI", "CLUSTER", "INSTRUMENT"],
            "HU_H": ["HU_NBT", "MGU", "HEADUNIT"],
            "RECODE": ["ENCODE", "PROGRAM", "CODE"]
        }

        prediction_upper = prediction.upper()

        for token in groundtruth_tokens:
            t_upper = token.upper()
            if t_upper in prediction_upper:
                return True
            if t_upper in EQUIV_MAP:
                for variant in EQUIV_MAP[t_upper]:
                    if variant in prediction_upper:
                        return True

        return False

    def _check_logic_conflict(self, groundtruth, prediction, field) -> bool:
        """Check for fatal technical contradictions."""
        if field == "RepairMeasures":
            prompt = f"""Role: BMW Master Technician (Workshop Foreman).
Task: Check for FATAL Technical Contradictions in Repair Measures.

[Ground Truth]: "{groundtruth}"
[Prediction]: "{prediction}"

Decision Rules:
1. Wrong Part: Different components -> CONFLICT
2. Wrong Action: Different actions (Program vs Replace) -> CONFLICT
3. Scope Expansion: Diagnosis-only but suggests Replace -> CONFLICT
   Exception: Conditional statements ("if faulty") or diagnostic pre-steps are SAFE

Reply: 'CONFLICT_DETECTED' or 'SAFE'"""

        else:  # FaultDescription
            prompt = f"""Role: BMW Diagnostic Specialist.
Task: Check for FATAL Diagnostic Contradictions in Fault Descriptions.

[Ground Truth]: "{groundtruth}"
[Prediction]: "{prediction}"

Decision Rules:
1. Physics Conflict: Opposite electrical states -> CONFLICT
2. Location Conflict: Different banks/sides -> CONFLICT
3. State Conflict: Open vs Short circuit -> CONFLICT
4. Synonyms: Similar terms are SAFE

Reply: 'CONFLICT_DETECTED' or 'SAFE'"""

        resp = self._call_llm_generic(self.api_url, self.api_key, self.model_name, prompt)
        return "CONFLICT_DETECTED" in resp.upper()

    def _normalize_to_dict(self, data: Union[Dict, List], keys: List[str]) -> Dict[str, str]:
        """Normalize data to dictionary with cleaned values."""
        junk_words = {"none", "n/a", "unknown", "null", "not specified", ""}

        if isinstance(data, dict):
            clean_data = {}
            for k, v in data.items():
                if v is None:
                    clean_data[k] = ""
                    continue
                val_str = str(v).strip()
                clean_data[k] = "" if val_str.lower() in junk_words else val_str
            for k in keys:
                if k not in clean_data:
                    clean_data[k] = ""
            return clean_data

        if isinstance(data, list):
            padded = data + [""] * (len(keys) - len(data))
            clean_dict = {}
            for k, v in zip(keys, padded):
                if v is None:
                    clean_dict[k] = ""
                    continue
                val_str = str(v).strip()
                clean_dict[k] = "" if val_str.lower() in junk_words else val_str
            return clean_dict

        return {k: "" for k in keys}

    def _extract_tuple_data(self, field, groundtruth, prediction):
        """Extract structured tuples for semantic analysis."""
        if field == "FaultDescription":
            keys = ["Fault_Anchor", "Location_Modifier", "Failure_State", "Code_Ref"]
            desc = "Extract: 1) Core component 2) Location 3) Failure mode 4) DTC codes"
        else:
            keys = ["Op_Action", "Target_Part", "Scope_Extent", "Prerequisite"]
            desc = "Extract: 1) Action verb 2) Target component 3) Scope 4) Conditions"

        prompt = f"""Task: Extract BMW Diagnostic Entities for semantic matching.
Field: {field}
Structure: {keys}
Definitions: {desc}

[groundtruth]: "{groundtruth}"
[prediction]: "{prediction}"

Return JSON with "groundtruth_Tuple" and "prediction_Tuple".
Normalize terms (e.g., 'Swap' -> 'Replace'). Use empty string for missing fields."""

        resp = self._call_llm_generic(self.api_url, self.api_key, self.model_name, prompt)
        result = {"groundtruth_Tuple": {}, "prediction_Tuple": {}}
        
        try:
            clean_resp = resp.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', clean_resp, re.DOTALL)
            json_str = match.group(0) if match else clean_resp

            try:
                raw = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    raw = ast.literal_eval(json_str)
                except:
                    print(f"      [Parsing Critical Fail] Could not parse JSON.")
                    raw = {}

            if isinstance(raw, dict):
                result["groundtruth_Tuple"] = self._normalize_to_dict(raw.get("groundtruth_Tuple", {}), keys)
                result["prediction_Tuple"] = self._normalize_to_dict(raw.get("prediction_Tuple", {}), keys)

        except Exception as e:
            print(f"      [Extraction Error] {e}")
        
        return result

    def _consult_search_llm(self, extraction_json):
        """Generate search hypotheses to validate semantic equivalence."""
        groundtruth_t = extraction_json.get("groundtruth_Tuple", {})
        prediction_t = extraction_json.get("prediction_Tuple", {})

        is_repair_mode = "Op_Action" in groundtruth_t or "Op_Action" in prediction_t
        query_hypotheses = []

        try:
            # Check component/part
            k_part = "Fault_Anchor" if "Fault_Anchor" in groundtruth_t else "Target_Part"
            v1 = str(groundtruth_t.get(k_part, "")).strip()
            v2 = str(prediction_t.get(k_part, "")).strip()
            if v1 and v2 and v1.lower() != v2.lower():
                query_hypotheses.append(
                    f"Search relation between BMW component '{v1}' and '{v2}'. Are they functionally equivalent?")

            # Check action/state
            k_act = "Failure_State" if "Failure_State" in groundtruth_t else "Op_Action"
            v1_act = str(groundtruth_t.get(k_act, "")).strip()
            v2_act = str(prediction_t.get(k_act, "")).strip()
            if v1_act and v2_act and v1_act.lower() != v2_act.lower():
                query_hypotheses.append(
                    f"In BMW ISTA, is '{v1_act}' synonymous with '{v2_act}'?")

            # Check location (fault description only)
            if "Location_Modifier" in groundtruth_t:
                l1 = str(groundtruth_t.get("Location_Modifier", "")).strip()
                l2 = str(prediction_t.get("Location_Modifier", "")).strip()
                if l1 and l2 and l1.lower() != l2.lower():
                    query_hypotheses.append(f"BMW Topology: Is '{l1}' same as '{l2}'?")

            # Check scope (repair measures only)
            if "Scope_Extent" in groundtruth_t:
                s1 = str(groundtruth_t.get("Scope_Extent", "")).strip()
                s2 = str(prediction_t.get("Scope_Extent", "")).strip()
                if s1 and s2 and s1.lower() != s2.lower():
                    query_hypotheses.append(f"Repair Scope: Does '{s1}' cover '{s2}'?")

            # Check prerequisites
            if "Prerequisite" in groundtruth_t:
                p1 = str(groundtruth_t.get("Prerequisite", "")).strip()
                p2 = str(prediction_t.get("Prerequisite", "")).strip()
                if p1 and p2 and p1.lower() != p2.lower():
                    query_hypotheses.append(f"Is condition '{p1}' equivalent to '{p2}'?")

        except Exception as e:
            print(f"[Arbiter Error] {e}")

        if not query_hypotheses: 
            return "No semantic conflicts detected."

        # Build context-specific rules
        if is_repair_mode:
            rule_instruction = """
**Repair Action Strictness:**
- Component equivalence: 'Wiring' == 'Harness', 'Post-cat sensor' == 'Sensor'
- Action precision: 'Replace' != 'Repair', 'Program' != 'Encode'
- Exception: 'Renew' == 'Replace', 'Swap' == 'Replace'"""
        else:
            rule_instruction = """
**Failure State Tolerance:**
- Generic states are equivalent: 'Implausible' == 'Failed' == 'Invalid'
- Exception: Physical contradictions (Short to Battery vs Ground) are conflicts"""

        prompt = f"""Role: BMW Senior Technical Specialist.
Task: Validate semantic equivalence of diagnostic terminology.

Hypotheses:
{chr(10).join([f"- {q}" for q in query_hypotheses])}

**Evaluation Rules:**
1. Part succession: FEM = BDC = CAS (generational equivalence)
2. Terminology synonyms: 'Renewal' = 'Replace', 'Line' = 'Wire'
3. Functional distinction: Different locations/components = CONFLICT
{rule_instruction}

Output JSON:
{{
    "analysis": "Brief justification",
    "equivalences": [
        {{ "hypothesis": "...", "is_equivalent": true/false, "reason": "..." }}
    ]
}}"""

        return self._call_llm_generic(self.same_api_url, self.same_api_key, self.same_model_name, prompt)

    def _build_final_rescue_prompt(self, field, groundtruth, prediction, logic_tag, tuples, evidence):
        """Build final arbitration prompt for expert decision."""
        if field == "FaultDescription":
            field_instruction = """
**VETERAN MECHANIC JUDGMENT (Lenient):**
1. Use arbitration evidence for component equivalence
2. Ignore verbosity differences
3. 'So What?' test: Same part being checked?
4. Failure mode distinction: Hardware fault != Sensor/logic issue
5. DTC override: Same fault code = PASS
6. Communication logic: 'No message' == 'Signal invalid'
7. General equivalence: 'Malfunction' == 'Failure' == 'Implausible'"""
        else:
            field_instruction = """
**OEM WARRANTY AUDIT (Strict):**
1. Scope expansion veto: Diagnosis-only -> Replace = FAIL
   Exception: Conditional ("if faulty") or pre-steps are SAFE
2. Specificity tolerance: Specific covers general
3. Arbitration check: Use equivalence validation
4. Critical omission: Missing power/voltage check = FAIL
5. Component logic: General inspection covers specific checks
6. Action severity: Action verbs must match"""

        return f"""Role: BMW Workshop Quality Control.
Trigger: {logic_tag}
Task: Final judgement for job card approval.

[Field]: {field}
[groundtruth]: "{groundtruth}"
[prediction]: "{prediction}"
[Structured Data]: {json.dumps(tuples, ensure_ascii=False)}
[Expert Arbitration]: {evidence}

{field_instruction}

Output (Reason in Chinese):
Reason: [Chinese explanation]
Conclusion: [PASS or FAIL]"""

    def llm_binary_judgment(self, field, groundtruth, prediction, granularity="Standard") -> Dict[str, Any]:
        """
        Main evaluation logic with multi-stage gates.
        
        Returns:
            Dict with judgment, score, similarity, logic path, and reason.
        """
        t_start = time.time()
        t_step1_end = t_start
        t_gate_end = t_start
        t_step2_end = t_start

        # Pre-check: Detect refusal/hallucination
        refusal_keywords = [
            "unable to obtain corresponding information from the network",
            "ai language model cannot predict"
        ]
        prediction_lower = prediction.lower()
        if any(k in prediction_lower for k in refusal_keywords):
            return {
                "judgment": "不符合",
                "score": 0,
                "sim": 0.0,
                "logic": "Pre-Check Fail (Refusal Detected)",
                "reason": "Prediction contains model refusal or invalid information.",
                "stage_timing": {"step1": 0, "gate": 0, "step2": 0}
            }

        # Compute similarity
        sim_val = self.compute_similarity(groundtruth, prediction)

        # Step 1: Get score from LLM
        score_resp = self._call_llm_generic(self.api_url, self.api_key, self.model_name,
                                            self._build_prompt_score(field, groundtruth, prediction))
        try:
            score = 0
            clean_text = score_resp.replace('*', '').replace('#', '').strip()

            # Parse score with multiple strategies
            m_label = re.search(r'(?:Score|Rating|Total)\D{0,15}[::]?\s*(\d+)', clean_text, re.IGNORECASE)
            m_frac = re.search(r'(\d+)\s*/\s*100', clean_text)
            m_end = re.search(r'(\d+)(?:\s*(?:pts|points))?\s*[.]?\s*$', clean_text, re.IGNORECASE)
            m_start = re.search(r'^\s*(\d+)', clean_text)

            if m_label:
                score = int(m_label.group(1))
            elif m_frac:
                score = int(m_frac.group(1))
            elif m_end:
                score = int(m_end.group(1))
            elif m_start:
                score = int(m_start.group(1))
            else:
                digits = re.findall(r'\d+', clean_text)
                if digits:
                    vals = [int(d) for d in digits]
                    valid_scores = [v for v in vals if 0 <= v <= 100]
                    if valid_scores:
                        last_val = valid_scores[-1]
                        max_val = max(valid_scores)
                        score = max_val if len(valid_scores) > 1 and last_val <= 10 and max_val >= 60 else last_val

            score = min(score, 100)
        except Exception as e:
            print(f"      [Score Parse Error] {e}")
            score = 0

        t_step1_end = time.time()

        # Compute hard entity matching
        groundtruth_has_hard_entity = self._check_hard_entities(groundtruth, groundtruth)
        prediction_has_match = self._check_hard_entities(groundtruth, prediction)

        # Gate 1: VIP Fast Track
        is_repair = (field == "RepairMeasures")
        vip_pass = False
        vip_reason = ""

        if score >= 90:
            if is_repair:
                # Check critical modifiers
                critical_mods = ["BANK", "CYLINDER", "LEFT", "RIGHT", "FRONT", "REAR", "INTAKE", "EXHAUST",
                                "UPPER", "LOWER", "INLET", "OUTLET", "UPSTREAM", "DOWNSTREAM", "SYSTEM"]
                groundtruth_u, prediction_u = groundtruth.upper(), prediction.upper()
                missing = any(m in groundtruth_u and m not in prediction_u for m in critical_mods)
                hallucinated = any(m in prediction_u and m not in groundtruth_u for m in critical_mods)
                logic_safe = not (missing or hallucinated)

                if score >= 98 and sim_val >= 0.75 and logic_safe:
                    vip_pass = True
                    vip_reason = "Repair VIP (Tier 1: Score 98+ & Sim>=0.75)"
                elif 90 <= score <= 97 and logic_safe and groundtruth_has_hard_entity and prediction_has_match:
                    vip_pass = True
                    vip_reason = "Repair VIP (Tier 2: Score 90-97 & Entity Match)"

            elif field == "FaultDescription":
                critical_mods = ["BANK", "CYLINDER", "LEFT", "RIGHT", "FRONT", "REAR", "INTAKE", "EXHAUST",
                                "UPPER", "LOWER", "INLET", "OUTLET", "UPSTREAM", "DOWNSTREAM",
                                "ENTIRE SYSTEM", "COMPLETE SYSTEM", "WHOLE SYSTEM"]
                groundtruth_u, prediction_u = groundtruth.upper(), prediction.upper()
                missing_mod = any(mod in groundtruth_u and mod not in prediction_u for mod in critical_mods)

                if not missing_mod:
                    if score >= 98 and sim_val >= 0.80:
                        vip_pass = True
                        vip_reason = "VIP Tier 1 (Score 98+ & Sim>=0.8)"
                    elif 90 <= score <= 97:
                        if groundtruth_has_hard_entity:
                            if prediction_has_match:
                                vip_pass = True
                                vip_reason = "VIP Tier 2 (Score 90-97 & Entity Match)"
                        else:
                            vip_pass = True
                            vip_reason = "VIP Tier 2 (Score 90-97 & High Precision)"

        elif not is_repair and sim_val >= 0.985:
            if groundtruth_has_hard_entity:
                if prediction_has_match:
                    vip_pass = True
                    vip_reason = "High Sim (0.985+) & Entity Verified"
            else:
                vip_pass = True
                vip_reason = "High Sim (0.985+)"

        # Gate 3: Logic conflict check
        should_check_gate3 = True
        if field == "FaultDescription" and score >= 80:
            should_check_gate3 = False

        if not vip_pass and score > 50 and sim_val < 0.99 and should_check_gate3:
            if self._check_logic_conflict(groundtruth, prediction, field):
                return {
                    "judgment": "不符合",
                    "score": score,
                    "sim": round(sim_val, 3),
                    "logic": "Gate 3 Fail (Logic Conflict)",
                    "reason": "Fatal repair logic contradiction detected.",
                    "stage_timing": {
                        "step1": t_step1_end - t_start,
                        "gate": time.time() - t_step1_end,
                        "step2": 0.0
                    }
                }

        t_gate_end = time.time()

        if vip_pass:
            return {
                "judgment": "符合",
                "score": score,
                "sim": round(sim_val, 3),
                "logic": f"Gate 1 VIP ({vip_reason})",
                "reason": "High confidence match, direct pass.",
                "stage_timing": {
                    "step1": t_step1_end - t_start,
                    "gate": t_gate_end - t_step1_end,
                    "step2": 0.0
                }
            }

        # Rescue mechanism
        rescue_triggered = False
        rescue_reason = ""

        if not vip_pass:
            if field == "FaultDescription":
                if 80 <= score <= 89:
                    rescue_triggered = True
                    rescue_reason = "Rescue Tier 3 (Score 80-89: Status Ambiguity)"
                elif 60 <= score <= 79:
                    rescue_triggered = True
                    rescue_reason = "Rescue Tier 4 (Score 60-79: Component Generalization)"
            else:  # RepairMeasures
                if 75 <= score <= 89:
                    rescue_triggered = True
                    rescue_reason = "Gate 2 (Repair Standard Rescue)"

            if not rescue_triggered:
                if (0.85 <= sim_val < 0.985):
                    if field != "RepairMeasures":
                        rescue_triggered = True
                        rescue_reason = f"Gate 2 (Sim {round(sim_val, 3)})"
                    elif sim_val > 0.95:
                        rescue_triggered = True
                        rescue_reason = f"Gate 2 (High Sim {round(sim_val, 3)})"

            if not rescue_triggered and prediction_has_match:
                rescue_triggered = True
                rescue_reason = "Gate 3 (Hard Entity/DTC Match)"

            if not rescue_triggered:
                if field == "RepairMeasures":
                    if (score >= 50) or (sim_val >= 0.80):
                        rescue_triggered = True
                        rescue_reason = "Gate 4 (Repair Safe Net)"
                else:
                    if (score >= 60) or (sim_val > 0.70):
                        rescue_triggered = True
                        rescue_reason = "Gate 4 (General Safe Net)"

        if rescue_triggered:
            # Step 2: Deep extraction and arbitration
            extracted_json = self._extract_tuple_data(field, groundtruth, prediction)
            
            groundtruth_t = extracted_json.get("groundtruth_Tuple", {})
            groundtruth_has_content = any(len(v) > 0 for v in groundtruth_t.values())

            if not groundtruth_has_content:
                t_step2_end = time.time()
                if field != "RepairMeasures" and score >= 60:
                    return {
                        "judgment": "符合",
                        "score": score,
                        "sim": round(sim_val, 3),
                        "logic": "Step 2 Bypass (Non-Technical groundtruth)",
                        "reason": "Groundtruth lacks technical entities, pass based on similarity.",
                        "stage_timing": {
                            "step1": t_step1_end - t_start,
                            "gate": t_gate_end - t_step1_end,
                            "step2": t_step2_end - t_gate_end
                        }
                    }
                else:
                    return {
                        "judgment": "不符合",
                        "score": score,
                        "sim": round(sim_val, 3),
                        "logic": "Step 2 Fail (Extraction Empty)",
                        "reason": "Structured extraction failed, cannot verify technical accuracy.",
                        "stage_timing": {
                            "step1": t_step1_end - t_start,
                            "gate": t_gate_end - t_step1_end,
                            "step2": t_step2_end - t_gate_end
                        }
                    }

            evidence = self._consult_search_llm(extracted_json)
            cot_resp = self._call_llm_generic(self.api_url, self.api_key, self.model_name,
                                              self._build_final_rescue_prompt(field, groundtruth, prediction,
                                                                              rescue_reason, extracted_json, evidence))

            # Parse conclusion
            is_pass = False
            resp_lower = cot_resp.lower()
            if "conclusion" in resp_lower:
                conc_part = resp_lower.split("conclusion")[-1]
                if any(k in conc_part for k in ["pass", "match", "correct"]) and "fail" not in conc_part:
                    is_pass = True
            elif any(k in resp_lower for k in ["pass", "match"]) and "fail" not in resp_lower:
                is_pass = True

            # Extract reason
            reason_text = "See logic."
            match = re.search(r'Reason[::]\s*(.*)', cot_resp, re.IGNORECASE)
            if match:
                r_raw = match.group(1)
                if "Conclusion" in r_raw:
                    reason_text = r_raw.split("Conclusion")[0].strip()
                elif "\n" in r_raw:
                    reason_text = r_raw.split("\n")[0].strip()
                else:
                    reason_text = r_raw.strip()

            t_step2_end = time.time()

            return {
                "judgment": "符合" if is_pass else "不符合",
                "score": score,
                "sim": round(sim_val, 3),
                "logic": f"{rescue_reason} -> {'Pass' if is_pass else 'Fail'}",
                "reason": reason_text,
                "stage_timing": {
                    "step1": t_step1_end - t_start,
                    "gate": t_gate_end - t_step1_end,
                    "step2": t_step2_end - t_gate_end
                }
            }

        # Final fail
        return {
            "judgment": "不符合",
            "score": score,
            "sim": round(sim_val, 3),
            "logic": "Step 1 Fail (Trash Bin)",
            "reason": f"Score ({score}) and similarity ({sim_val:.2f}) too low, no entity match.",
            "stage_timing": {
                "step1": t_step1_end - t_start,
                "gate": t_gate_end - t_step1_end,
                "step2": 0.0
            }
        }
