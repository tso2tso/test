"""
Data Preprocessing Module

Pipeline:
1. Load raw Integrated_Data.json from project root
2. Balance long-tail distribution (downsample high-frequency RepairMeasures)
3. Build SFT training format
4. Split into train/val sets

"""

import json
import os
import sys
from collections import Counter
from collections import deque
from typing import Dict, List, Tuple
import random
from tqdm import tqdm

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, DataProcessingConfig


def _tail_non_whitespace(path: str, max_bytes: int = 8192) -> str:
    """Read tail of file to check for truncation."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(size, max_bytes)
            f.seek(-read_size, os.SEEK_END)
            chunk = f.read(read_size)
        text = chunk.decode("utf-8", errors="replace")
        return text.rstrip()
    except Exception:
        return ""


def _print_json_error_context(path: str, lineno: int, colno: int, radius: int = 3) -> None:
    """Print context around JSON parse error for debugging."""
    if lineno <= 0:
        return

    start = max(1, lineno - radius)
    end = lineno + radius
    buf: deque[tuple[int, str]] = deque()

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if i < start:
                    continue
                if i > end:
                    break
                buf.append((i, line.rstrip("\n")))
    except Exception as ex:
        print(f"[Error] Cannot read context: {ex}")
        return

    print("\n[JSON Decode Error] Context around error:")
    for i, line in buf:
        prefix = ">>" if i == lineno else "  "
        print(f"{prefix} L{i}: {line}")
        if i == lineno and colno > 0:
            pointer = " " * (len(f"{prefix} L{i}: ") + max(0, colno - 1)) + "^"
            print(pointer)


def load_raw_data(file_path: str) -> List[Dict]:
    """Load raw data from JSON file."""
    print(f"Loading raw data: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"\n[Error] Invalid JSON format.")
        print(f"- Position: line={e.lineno}, column={e.colno}")
        print(f"- Message: {e.msg}")

        _print_json_error_context(file_path, e.lineno, e.colno, radius=4)

        tail = _tail_non_whitespace(file_path)
        if tail and (not tail.endswith("]")):
            print("\n[Hint] File may be truncated (missing closing ']').")

        raise
    print(f"Loaded {len(data)} records")
    return data


def analyze_distribution(data: List[Dict]) -> Dict[str, int]:
    """Analyze RepairMeasures distribution."""
    Repair_measures_counter = Counter()
    
    for item in data:
        output = item.get("output", {})
        if isinstance(output, dict):
            Repair_measures = output.get("RepairMeasures", "")
        else:
            Repair_measures = str(output)
        Repair_measures_counter[Repair_measures] += 1
    
    return dict(Repair_measures_counter)


def balance_data(
    data: List[Dict], 
    config: DataProcessingConfig
) -> List[Dict]:
    """
    Balance long-tail distribution using square root sampling.
    
    Strategy:
    - High frequency: downsample to sqrt(n) * coefficient
    - Low frequency: keep all samples
    """
    import math
    
    print("\nStarting data balancing (square root sampling)...")
    
    # Group by RepairMeasures
    grouped_data: Dict[str, List[Dict]] = {}
    for item in data:
        output = item.get("output", {})
        if isinstance(output, dict):
            Repair_measures = output.get("RepairMeasures", "")
        else:
            Repair_measures = str(output)
        
        if Repair_measures not in grouped_data:
            grouped_data[Repair_measures] = []
        grouped_data[Repair_measures].append(item)
    
    print(f"Found {len(grouped_data)} unique RepairMeasures")
    
    # Analyze original distribution
    counts = [len(items) for items in grouped_data.values()]
    print(f"Original distribution - Max: {max(counts)}, Min: {min(counts)}, Avg: {sum(counts)/len(counts):.1f}")
    
    sqrt_coefficient = getattr(config, 'sqrt_sampling_coefficient', 10)
    
    # Balance processing
    balanced_data = []
    random.seed(config.seed)
    
    sampling_stats = {"downsampled": 0, "kept_all": 0}
    
    for Repair_measures, items in tqdm(grouped_data.items(), desc="Square root sampling"):
        n = len(items)
        target_n = max(
            config.min_samples_per_response,
            min(int(math.sqrt(n) * sqrt_coefficient), n)
        )
        
        if target_n < n:
            sampled = random.sample(items, target_n)
            balanced_data.extend(sampled)
            sampling_stats["downsampled"] += 1
        else:
            balanced_data.extend(items)
            sampling_stats["kept_all"] += 1
    
    print(f"Sampling stats - Downsampled: {sampling_stats['downsampled']}, Kept all: {sampling_stats['kept_all']}")
    print(f"Balanced size: {len(balanced_data)} (Original: {len(data)}, Ratio: {len(balanced_data)/len(data)*100:.1f}%)")
    
    # Calculate class weights for SM and FD
    sm_weights = {}
    for Repair_measures, items in grouped_data.items():
        sm_weights[Repair_measures] = math.sqrt(len(data) / len(items))
    
    fd_grouped: Dict[str, List[Dict]] = {}
    for item in data:
        output = item.get("output", {})
        if isinstance(output, dict):
            fault_desc = output.get("FaultDescription", "")
        else:
            fault_desc = str(output)
        
        if fault_desc not in fd_grouped:
            fd_grouped[fault_desc] = []
        fd_grouped[fault_desc].append(item)
    
    fd_weights = {}
    for fault_desc, items in fd_grouped.items():
        fd_weights[fault_desc] = math.sqrt(len(data) / len(items))
    
    print(f"\n=== Class Weights Statistics ===")
    print(f"SM classes: {len(sm_weights)}, Weight range: [{min(sm_weights.values()):.2f}, {max(sm_weights.values()):.2f}]")
    print(f"FD classes: {len(fd_weights)}, Weight range: [{min(fd_weights.values()):.2f}, {max(fd_weights.values()):.2f}]")
    
    # Save weights to processed directory
    weights_dir = DATA_CONFIG["processed_data_dir"]
    os.makedirs(weights_dir, exist_ok=True)
    
    sm_weights_path = os.path.join(weights_dir, "sm_weights.json")
    with open(sm_weights_path, "w", encoding="utf-8") as f:
        json.dump(sm_weights, f, ensure_ascii=False, indent=2)
    print(f"SM weights saved to: {sm_weights_path}")
    
    fd_weights_path = os.path.join(weights_dir, "fd_weights.json")
    with open(fd_weights_path, "w", encoding="utf-8") as f:
        json.dump(fd_weights, f, ensure_ascii=False, indent=2)
    print(f"FD weights saved to: {fd_weights_path}")
    
    class_weights = {
        "sm_weights": sm_weights,
        "fd_weights": fd_weights,
    }
    combined_path = os.path.join(weights_dir, "class_weights.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(class_weights, f, ensure_ascii=False, indent=2)
    print(f"Combined weights saved to: {combined_path}")
    
    return balanced_data


def parse_input(input_str: str) -> Dict[str, str]:
    """
    Parse input string to extract ECUID, DTC, Trigger, TimeCondition.
    Input format: "ECUID: 0087, DTC: 5D0C, Trigger: A fault..., TimeCondition: 5 s"
    """
    result = {
        "ecuid": "",
        "dtc": "",
        "trigger": "",
        "time_condition": ""
    }
    
    parts = input_str.split(", ")
    for part in parts:
        if part.startswith("ECUID:"):
            result["ecuid"] = part.replace("ECUID:", "").strip()
        elif part.startswith("DTC:"):
            result["dtc"] = part.replace("DTC:", "").strip()
        elif part.startswith("Trigger:"):
            result["trigger"] = part.replace("Trigger:", "").strip()
        elif part.startswith("TimeCondition:"):
            result["time_condition"] = part.replace("TimeCondition:", "").strip()
    
    return result


def format_for_sft(
    data: List[Dict], 
    config: DataProcessingConfig
) -> List[Dict]:
    """
    Convert data to SFT training format.
    Format: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    print("\nConverting to SFT format...")
    
    formatted_data = []
    
    for item in tqdm(data, desc="Format conversion"):
        input_str = item.get("input", "")
        output = item.get("output", {})
        
        parsed_input = parse_input(input_str)
        
        user_content = config.user_template.format(
            ecuid=parsed_input["ecuid"],
            dtc=parsed_input["dtc"],
            trigger=parsed_input["trigger"],
            time_condition=parsed_input["time_condition"]
        )
        
        if isinstance(output, dict):
            assistant_content = json.dumps(output, ensure_ascii=False, indent=2)
        else:
            assistant_content = json.dumps({
                "FaultDescription": str(output),
                "RepairMeasures": ""
            }, ensure_ascii=False, indent=2)
        
        formatted_item = {
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "metadata": {
                "original_input": input_str,
                "ecuid": parsed_input["ecuid"],
                "dtc": parsed_input["dtc"],
            }
        }
        
        formatted_data.append(formatted_item)
    
    print(f"Conversion complete: {len(formatted_data)} records")
    return formatted_data


def split_data(
    data: List[Dict], 
    val_ratio: float, 
    seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """Split data into train and validation sets."""
    print(f"\nSplitting dataset (val ratio: {val_ratio})...")
    
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    val_size = int(len(shuffled) * val_ratio)
    val_data = shuffled[:val_size]
    train_data = shuffled[val_size:]
    
    print(f"Train set: {len(train_data)} records")
    print(f"Val set: {len(val_data)} records")
    
    return train_data, val_data


def save_data(data: List[Dict], file_path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Data saved to: {file_path}")

def get_item_key(item: Dict) -> str:
    """Obtain the unique identifier of the sample"""
    input_str = item.get("input", "")
    output = item.get("output", {})
    if isinstance(output, dict):
        sm = output.get("RepairMeasures", "")
    else:
        sm = str(output)
    return f"{input_str}|||{sm}"


def create_test_benchmark(
    raw_data: List[Dict],
    balanced_data: List[Dict],
    config: DataProcessingConfig,
    total_samples: int = 100,
    head_ratio: float = 0.3,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build test benchmark dataset.
    
    Strategy:
    1. Sample from high-frequency items cut by balancing (unseen by model) -> test generalization
    2. Sample from long-tail items in sft_balanced (rare classes) -> test learning
    
    Returns:
        Tuple: (formatted_data, raw_data)
    """
    import math
    
    print("\n" + "=" * 60)
    print("=== Building Test Benchmark ===")
    print("=" * 60)
    
    random.seed(config.seed + 999)
    
    # Step 1: Find items cut by balancing
    balanced_keys = set(get_item_key(item) for item in balanced_data)
    cut_samples = [item for item in raw_data if get_item_key(item) not in balanced_keys]
    print(f"Cut high-frequency samples: {len(cut_samples)}")
    
    # Group cut samples by RepairMeasures
    cut_by_sm: Dict[str, List[Dict]] = {}
    for item in cut_samples:
        output = item.get("output", {})
        sm = output.get("RepairMeasures", "") if isinstance(output, dict) else str(output)
        if sm not in cut_by_sm:
            cut_by_sm[sm] = []
        cut_by_sm[sm].append(item)
    
    # Step 2: Sample from cut high-frequency items
    head_samples_count = int(total_samples * head_ratio)
    head_benchmark = []
    
    cut_sm_keys = list(cut_by_sm.keys())
    random.shuffle(cut_sm_keys)
    
    for sm in cut_sm_keys:
        if len(head_benchmark) >= head_samples_count:
            break
        head_benchmark.append(random.choice(cut_by_sm[sm]))
    
    if len(head_benchmark) < head_samples_count:
        remaining = head_samples_count - len(head_benchmark)
        all_cut = [item for items in cut_by_sm.values() for item in items]
        head_keys = set(get_item_key(item) for item in head_benchmark)
        available = [item for item in all_cut if get_item_key(item) not in head_keys]
        if available:
            head_benchmark.extend(random.sample(available, min(remaining, len(available))))
    
    print(f"Sampled from high-frequency cut: {len(head_benchmark)} records")
    
    # Step 3: Sample long-tail items from sft_balanced
    tail_samples_count = total_samples - len(head_benchmark)
    
    balanced_by_sm: Dict[str, List[Dict]] = {}
    for item in balanced_data:
        output = item.get("output", {})
        sm = output.get("RepairMeasures", "") if isinstance(output, dict) else str(output)
        if sm not in balanced_by_sm:
            balanced_by_sm[sm] = []
        balanced_by_sm[sm].append(item)
    
    sm_by_count = sorted(balanced_by_sm.items(), key=lambda x: len(x[1]))
    
    tail_benchmark = []
    used_sm = set()
    
    for sm, items in sm_by_count:
        if len(tail_benchmark) >= tail_samples_count:
            break
        tail_benchmark.append(random.choice(items))
        used_sm.add(sm)
    
    round_num = 2
    while len(tail_benchmark) < tail_samples_count and round_num <= 5:
        for sm, items in sm_by_count:
            if len(tail_benchmark) >= tail_samples_count:
                break
            if len(items) >= round_num:
                available = [item for item in items if get_item_key(item) not in 
                            set(get_item_key(t) for t in tail_benchmark)]
                if available:
                    tail_benchmark.append(random.choice(available))
        round_num += 1
    
    print(f"Sampled from SFT long-tail: {len(tail_benchmark)} records, covering {len(used_sm)} classes")
    
    # Step 4: Merge and format
    benchmark_raw = head_benchmark + tail_benchmark
    random.shuffle(benchmark_raw)
    
    benchmark_formatted = format_for_sft(benchmark_raw, config)
    
    for i, item in enumerate(benchmark_formatted):
        if i < len(head_benchmark):
            item["metadata"]["source"] = "head_unseen"
        else:
            item["metadata"]["source"] = "tail_sft"
    
    # Step 5: Statistics
    print(f"\n📊 Benchmark Statistics:")
    print(f"   - Total samples: {len(benchmark_formatted)}")
    print(f"   - High-frequency unseen (head_unseen): {len(head_benchmark)} ({len(head_benchmark)/len(benchmark_formatted)*100:.1f}%)")
    print(f"   - Long-tail SFT (tail_sft): {len(tail_benchmark)} ({len(tail_benchmark)/len(benchmark_formatted)*100:.1f}%)")
    
    all_sm = set()
    for item in benchmark_raw:
        output = item.get("output", {})
        sm = output.get("RepairMeasures", "") if isinstance(output, dict) else str(output)
        all_sm.add(sm)
    print(f"   - Output class coverage: {len(all_sm)}")
    
    return benchmark_formatted, benchmark_raw


def main():
    """
    Main function - Two-stage data strategy
    
    1. SFT Stage: Use balanced data (square root sampling)
       - Purpose: Learn output space quickly
       - Smaller dataset, faster training
       
    2. Distillation Stage: Use full data
       - Purpose: Learn all condition-output mappings
       - Teacher signal aligns input-output
    """
    config = DataProcessingConfig()
    
    # 1. Load raw data (from project root)
    raw_data = load_raw_data(DATA_CONFIG["raw_data_path"])
    
    # 2. Analyze distribution
    print("\n=== Original Data Distribution ===")
    distribution = analyze_distribution(raw_data)
    sorted_dist = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 High-frequency RepairMeasures:")
    for i, (measures, count) in enumerate(sorted_dist[:10]):
        print(f"  {i+1}. [{count}] {measures[:80]}...")
    
    # 3. Balance data (for SFT)
    balanced_raw_data = balance_data(raw_data, config)
    
    # 4. Build test benchmark (must do this BEFORE filtering training data)
    benchmark_data, benchmark_raw = create_test_benchmark(
        raw_data=raw_data,
        balanced_data=balanced_raw_data,
        config=config,
        total_samples=100,
        head_ratio=0.3,  
    )
    
    # 5. Filter out benchmark samples from training data
    print(f"\n[Data Filtering] Removing {len(benchmark_raw)} benchmark samples from training data...")
    benchmark_keys = set(get_item_key(item) for item in benchmark_raw)
    train_raw_data = [item for item in raw_data if get_item_key(item) not in benchmark_keys]
    sft_raw_data = [item for item in balanced_raw_data if get_item_key(item) not in benchmark_keys]
    
    print(f"Original data: {len(raw_data)} -> Filtered: {len(train_raw_data)} (removed {len(raw_data) - len(train_raw_data)})")
    print(f"Balanced data: {len(balanced_raw_data)} -> Filtered: {len(sft_raw_data)} (removed {len(balanced_raw_data) - len(sft_raw_data)})")
    
    # 6. Convert to SFT format
    print("\n[Full Data] Converting format...")
    full_formatted_data = format_for_sft(train_raw_data, config)
    
    print("\n[Balanced Data] Converting format...")
    balanced_formatted_data = format_for_sft(sft_raw_data, config)
    
    # 7. Split validation set
    _, val_data = split_data(balanced_formatted_data, config.val_ratio, config.seed)
    
    # 8. Save files
    save_data(balanced_formatted_data, DATA_CONFIG["sft_data_path"])
    save_data(full_formatted_data, DATA_CONFIG["train_data_path"])
    save_data(val_data, DATA_CONFIG["val_data_path"])
    
    benchmark_path = os.path.join(DATA_CONFIG["processed_data_dir"], "test_benchmark.json")
    save_data(benchmark_data, benchmark_path)
    
    # 9. Print summary
    print("\n" + "=" * 60)
    print("=== Data Processing Complete ===")
    print("=" * 60)
    print(f"\nSFT Data (Balanced):")
    print(f"   - File: sft_balanced.json")
    print(f"   - Size: {len(balanced_formatted_data)} records")
    
    print(f"\nDistillation Data (Full):")
    print(f"   - File: train.json")
    print(f"   - Size: {len(full_formatted_data)} records")
    
    print(f"\nValidation Set:")
    print(f"   - File: val.json")
    print(f"   - Size: {len(val_data)} records")
    
    print(f"\nTest Benchmark:")
    print(f"   - File: test_benchmark.json")
    print(f"   - Size: {len(benchmark_data)} records")
    
    # Check balanced distribution
    print("\n=== Balanced Distribution ===")
    balanced_dist = analyze_distribution(balanced_raw_data)
    sorted_balanced = sorted(balanced_dist.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 RepairMeasures (Balanced):")
    for i, (measures, count) in enumerate(sorted_balanced[:10]):
        print(f"  {i+1}. [{count}] {measures[:80]}...")
    
    print("\n=== Condition Coverage ===")
    print(f"Full data: {len(raw_data)} unique conditions")
    print(f"Balanced data: {len(balanced_raw_data)} unique conditions")
    print(f"Coverage: {len(balanced_raw_data)/len(raw_data)*100:.1f}%")


if __name__ == "__main__":
    main()
