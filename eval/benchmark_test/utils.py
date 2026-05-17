"""
Utility Functions Module
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, List


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data: Any, file_path: str) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        file_path: File path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Data saved to: {file_path}")


def load_json(file_path: str) -> Any:
    """
    Load JSON file.
    
    Args:
        file_path: File path
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_test_benchmark(data_path: str) -> List[Dict[str, Any]]:
    """
    Load test_benchmark.json format test data.
    
    Args:
        data_path: Data file path
        
    Returns:
        List of test cases
    """
    print(f"[DataLoader] Loading data: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_cases = []
    
    for item in data:
        messages = item.get("messages", [])
        metadata = item.get("metadata", {})
        
        # Extract user input
        user_content = ""
        assistant_content = ""
        
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]
        
        # Parse user input
        input_data = parse_user_input(user_content)
        
        # Parse assistant output as groundtruth
        groundtruth = parse_assistant_output(assistant_content)
        
        test_cases.append({
            "input": input_data,
            "groundtruth": groundtruth,
            "metadata": metadata
        })
    
    print(f"[DataLoader] ✅ Loaded {len(test_cases)} test cases")
    return test_cases


def parse_user_input(content: str) -> Dict[str, str]:
    """
    Parse user input content.
    
    Args:
        content: User input string
        
    Returns:
        Parsed input dictionary
    """
    result = {
        "ECU_ID": "",
        "DTC": "",
        "Trigger": "",
        "TimeCondition": ""
    }
    
    # Split by comma
    parts = content.split(",")
    
    for part in parts:
        part = part.strip()
        if part.startswith("ECUID:"):
            result["ECU_ID"] = part.split(":", 1)[1].strip()
        elif part.startswith("DTC:"):
            result["DTC"] = part.split(":", 1)[1].strip()
        elif part.startswith("Trigger:"):
            result["Trigger"] = part.split(":", 1)[1].strip()
        elif part.startswith("TimeCondition:"):
            result["TimeCondition"] = part.split(":", 1)[1].strip()
    
    return result


def parse_assistant_output(content: str) -> Dict[str, str]:
    """
    Parse assistant output content.
    
    Args:
        content: Assistant output string
        
    Returns:
        Parsed groundtruth dictionary
    """
    result = {
        "FaultDescription": "",
        "RepairMeasures": ""
    }
    
    try:
        # Try to parse JSON
        parsed = json.loads(content)
        result["FaultDescription"] = parsed.get("FaultDescription", "")
        result["RepairMeasures"] = parsed.get("RepairMeasures", parsed.get("ServiceMeasures", ""))
    except json.JSONDecodeError:
        # If parsing fails, try to extract JSON portion
        try:
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                parsed = json.loads(json_str)
                result["FaultDescription"] = parsed.get("FaultDescription", "")
                result["RepairMeasures"] = parsed.get("RepairMeasures", parsed.get("ServiceMeasures", ""))
        except:
            # Last resort: use original content
            result["RepairMeasures"] = content
    
    return result


def sample_test_cases(test_cases: List[Dict[str, Any]], n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Randomly sample test cases.
    
    Args:
        test_cases: List of test cases
        n: Sample size
        seed: Random seed
        
    Returns:
        Sampled test cases list
    """
    import random
    random.seed(seed)
    
    if n >= len(test_cases):
        return test_cases
    
    return random.sample(test_cases, n)


def save_prediction_text(records: List[Dict[str, Any]], save_path: str) -> None:
    """
    Save model prediction records as readable text file.
    
    Args:
        records: Prediction record list
        save_path: Save path
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Model Prediction Output Complete Record\n")
        f.write("=" * 120 + "\n\n")
        
        for record in records:
            idx = record.get("index", "?")
            input_data = record.get("input", {})
            raw_data = record.get("raw_output", {})
            
            f.write(f"[Sample {idx}]\n")
            f.write("-" * 100 + "\n")
            
            f.write("Input info:\n")
            for key, value in input_data.items():
                f.write(f"  • {key}: {value}\n")
            
            f.write("\nModel raw output:\n")
            if isinstance(raw_data, dict):
                f.write(f"  {json.dumps(raw_data, ensure_ascii=False, indent=2)}\n")
            else:
                f.write(f"  {raw_data}\n")
            
            f.write("\n" + "=" * 120 + "\n\n")
    
    print(f"✅ Prediction text saved to: {save_path}")


class BenchmarkDataLoader:
    """Benchmark data loader"""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader.
        
        Args:
            data_path: Data file path
        """
        self.data_path = data_path
        self.data = None
        self.test_cases = None
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load raw data"""
        if self.data is None:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        return self.data
    
    def get_test_cases(self, n: int = None) -> List[Dict[str, Any]]:
        """
        Get test cases.
        
        Args:
            n: Number of test cases to get, None for all
            
        Returns:
            List of test cases
        """
        if self.test_cases is None:
            self.test_cases = load_test_benchmark(self.data_path)
        
        if n is None or n >= len(self.test_cases):
            return self.test_cases
        
        return sample_test_cases(self.test_cases, n)
