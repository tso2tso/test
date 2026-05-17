"""
Multi-Pair Accuracy Test Script
Test model prediction accuracy in multi-pair scenarios
Each test group contains multiple input pairs, all must pass for the group to pass
Supports local fine-tuned model inference
"""
import os
import sys
import json
import argparse
from typing import Dict, Any, List

# Get current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add project paths (ensure current directory has priority)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Use explicit import to avoid config conflict with root directory
import importlib.util
spec = importlib.util.spec_from_file_location("inference_config", os.path.join(SCRIPT_DIR, "config.py"))
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

from utils import save_json, get_timestamp, BenchmarkDataLoader, save_prediction_text
from local_model_interface import LocalModelInterface

# Import evaluator (unified multiagent_evaluator)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "eval"))
from evaluators import ThresholdCalibrationEvaluator


def run_multi_pair_predictions(
    model: LocalModelInterface,
    test_cases_per_group: List[List[Dict[str, Any]]],
    temperature: float = 0.7
) -> tuple:
    """
    Run multi-pair predictions
    
    Args:
        model: Model interface
        test_cases_per_group: Test cases grouped by test group
        temperature: Sampling temperature
        
    Returns:
        (predictions list, raw output records)
    """
    all_predictions = []
    all_raw_outputs = []
    
    print(f"\n{'='*80}")
    print(f"Starting Multi-Pair Predictions")
    print(f"{'='*80}")
    print(f"Number of test groups: {len(test_cases_per_group)}")
    
    for group_idx, test_cases in enumerate(test_cases_per_group):
        print(f"\n--- Test Group {group_idx + 1}/{len(test_cases_per_group)} ---")
        print(f"Number of pairs in this group: {len(test_cases)}")
        
        group_predictions = []
        group_raw_outputs = []
        
        for pair_idx, test_case in enumerate(test_cases):
            print(f"  Pair {pair_idx + 1}/{len(test_cases)}...")
            input_data = test_case["input"]
            output_data = test_case["groundtruth"]
            
            # Predict
            prediction, raw_data = model.predict(
                ecu_id=input_data["ECU_ID"],
                dtc=input_data["DTC"],
                trigger=input_data["Trigger"],
                timecondition=input_data["TimeCondition"],
                temperature=temperature
            )
            
            prediction["FaultDescription_gt"] = output_data["FaultDescription"]
            prediction["RepairMeasures_gt"] = output_data["RepairMeasures"]
            
            group_predictions.append(prediction)
            group_raw_outputs.append({
                "index": f"{group_idx + 1}-{pair_idx + 1}",
                "input": input_data,
                "raw_output": raw_data
            })
            
            print(f"    Done")
        
        all_predictions.append(group_predictions)
        all_raw_outputs.extend(group_raw_outputs)
    
    return all_predictions, all_raw_outputs


def run_multi_pair_evaluation(
    predictions_per_group: List[List[Dict[str, str]]],
    groundtruths_per_group: List[List[Dict[str, str]]],
    model: LocalModelInterface = None
) -> Dict[str, Any]:
    """
    Run multi-pair evaluation
    
    Args:
        predictions_per_group: Predictions grouped by test group
        groundtruths_per_group: Ground truths grouped by test group
        model: Model interface (for retry)
        
    Returns:
        Evaluation results
    """
    print("\n" + "=" * 80)
    print("Starting Multi-Pair Accuracy Evaluation")
    print("=" * 80)
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    print(f"  LLM Model: {config.LLM_EVALUATOR_CONFIG['model_name']}")
    
    evaluator = ThresholdCalibrationEvaluator(
        semantic_model_name=config.get_semantic_model_path(),
        llm_api_url=config.LLM_EVALUATOR_CONFIG["api_url"],
        llm_api_key=config.LLM_EVALUATOR_CONFIG["api_key"],
        llm_model_name=config.LLM_EVALUATOR_CONFIG["model_name"],
        same_config=config.LLM_SAME_CONFIG
    )
    
    # Evaluate each group
    all_group_results = []
    total_pass = 0
    total_fail = 0
    
    for group_idx, (preds, gts) in enumerate(zip(predictions_per_group, groundtruths_per_group)):
        print(f"\n{'='*60}")
        print(f"Evaluating Test Group {group_idx + 1}/{len(predictions_per_group)}")
        print(f"{'='*60}")
        
        group_result = {
            "group_index": group_idx,
            "num_pairs": len(preds),
            "pair_results": [],
            "all_pass": True
        }
        
        for pair_idx, (pred, gt) in enumerate(zip(preds, gts)):
            print(f"\n  Pair {pair_idx + 1}/{len(preds)}:")
            
            pair_result = {
                "pair_index": pair_idx,
                "FaultDescription": {},
                "RepairMeasures": {}
            }
            
            # Evaluate FaultDescription
            fault_pred = pred.get("FaultDescription", "")
            fault_gt = gt.get("FaultDescription", "")
            
            print(f"    Evaluating FaultDescription...")
            fault_judgment = evaluator.llm_binary_judgment("FaultDescription", fault_gt, fault_pred)
            
            # Retry mechanism
            if fault_judgment["judgment"] != "符合" and model is not None:
                for retry in range(2):
                    print(f"      FaultDescription mismatch, retry {retry + 1}/2...")
                    new_pred = model.re_predict_faultdescription(pred)
                    fault_pred = new_pred.get("FaultDescription", "")
                    fault_judgment = evaluator.llm_binary_judgment("FaultDescription", fault_gt, fault_pred)
                    if fault_judgment["judgment"] == "符合":
                        break
            
            pair_result["FaultDescription"] = {
                "prediction": fault_pred,
                "groundtruth": fault_gt,
                "judgment": fault_judgment["judgment"],
                "score": fault_judgment.get("score", 0),
                "pass": fault_judgment["judgment"] == "符合"
            }
            
            print(f"      Result: {fault_judgment['judgment']} (Score: {fault_judgment.get('score', 0)})")
            
            # Evaluate RepairMeasures
            repair_pred = pred.get("RepairMeasures", "")
            repair_gt = gt.get("RepairMeasures", "")
            
            print(f"    Evaluating RepairMeasures...")
            repair_judgment = evaluator.llm_binary_judgment("RepairMeasures", repair_gt, repair_pred)
            
            # Retry mechanism
            if repair_judgment["judgment"] != "符合" and model is not None:
                for retry in range(2):
                    print(f"      RepairMeasures mismatch, retry {retry + 1}/2...")
                    new_pred = model.re_predict_repairmeasures(pred, fault_judgment["judgment"])
                    repair_pred = new_pred.get("RepairMeasures", "")
                    repair_judgment = evaluator.llm_binary_judgment("RepairMeasures", repair_gt, repair_pred)
                    if repair_judgment["judgment"] == "符合":
                        break
            
            pair_result["RepairMeasures"] = {
                "prediction": repair_pred,
                "groundtruth": repair_gt,
                "judgment": repair_judgment["judgment"],
                "score": repair_judgment.get("score", 0),
                "pass": repair_judgment["judgment"] == "符合"
            }
            
            print(f"      Result: {repair_judgment['judgment']} (Score: {repair_judgment.get('score', 0)})")
            
            # Check if this pair passed
            pair_pass = pair_result["FaultDescription"]["pass"] and pair_result["RepairMeasures"]["pass"]
            pair_result["pass"] = pair_pass
            
            if not pair_pass:
                group_result["all_pass"] = False
            
            group_result["pair_results"].append(pair_result)
        
        # Count group pass/fail
        if group_result["all_pass"]:
            total_pass += 1
            print(f"\n  Test Group {group_idx + 1} ALL PASSED")
        else:
            total_fail += 1
            print(f"\n  Test Group {group_idx + 1} NOT ALL PASSED")
        
        all_group_results.append(group_result)
    
    # Calculate pair-level pass rate (all passed pairs / total pairs)
    total_pairs = 0
    passed_pairs = 0
    fault_passed = 0
    repair_passed = 0
    
    for group in all_group_results:
        for pair in group["pair_results"]:
            total_pairs += 1
            if pair["pass"]:
                passed_pairs += 1
            if pair["FaultDescription"]["pass"]:
                fault_passed += 1
            if pair["RepairMeasures"]["pass"]:
                repair_passed += 1
    
    # Statistics
    statistics = {
        "total_groups": len(predictions_per_group),
        "pass_groups": total_pass,
        "fail_groups": total_fail,
        "group_pass_rate": total_pass / len(predictions_per_group) if predictions_per_group else 0,
        # Pair-level statistics
        "pair_level_statistics": {
            "total_pairs": total_pairs,
            "passed_pairs": passed_pairs,
            "pair_pass_rate": passed_pairs / total_pairs if total_pairs > 0 else 0,
            "fault_passed": fault_passed,
            "fault_pass_rate": fault_passed / total_pairs if total_pairs > 0 else 0,
            "repair_passed": repair_passed,
            "repair_pass_rate": repair_passed / total_pairs if total_pairs > 0 else 0,
        }
    }
    
    return {
        "statistics": statistics,
        "all_groups": all_group_results
    }


def print_summary(results: Dict[str, Any], metadata: Dict[str, Any]):
    """Print test summary"""
    stats = results["statistics"]
    
    print("\n" + "=" * 80)
    print("Multi-Pair Accuracy Test Summary")
    print("=" * 80)
    
    print(f"\nTest Configuration:")
    print(f"  Total Test Groups: {stats['total_groups']}")
    print(f"  Model: {metadata.get('model_path', 'Unknown')}")
    print(f"  Evaluator: {metadata.get('llm_evaluator', 'Unknown')}")
    
    # Pair-level pass rate (core metric)
    pls = stats.get("pair_level_statistics", {})
    if pls:
        print(f"\n{'='*40}")
        print(f"Pair-Level Pass Rate")
        print(f"{'='*40}")
        print(f"  Total Pairs: {pls['total_pairs']}")
        print(f"  Passed Pairs: {pls['passed_pairs']}")
        print(f"  Pair Pass Rate: {pls['pair_pass_rate']:.2%} ({pls['passed_pairs']}/{pls['total_pairs']})")
        print(f"  ")
        print(f"  FaultDescription Passed: {pls['fault_passed']}/{pls['total_pairs']} ({pls['fault_pass_rate']:.2%})")
        print(f"  RepairMeasures Passed: {pls['repair_passed']}/{pls['total_pairs']} ({pls['repair_pass_rate']:.2%})")
    
    print(f"\nGroup-Level Results (all pairs must pass for group to pass):")
    print(f"  Passed Groups: {stats['pass_groups']}")
    print(f"  Failed Groups: {stats['fail_groups']}")
    print(f"  Group Pass Rate: {stats['group_pass_rate']:.2%}")
    
    print("\n" + "=" * 80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run multi-pair accuracy test (using local fine-tuned model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run new multi-pair prediction test
  python run_multi_pair_accuracy_test.py --run-prediction --num-groups 5 --pairs-per-group 3
  
  # Only generate predictions, skip evaluation
  python run_multi_pair_accuracy_test.py --run-prediction --skip-eval
  
  # Use existing prediction results for evaluation
  python run_multi_pair_accuracy_test.py --from-file results/xxx/predictions.json
        """
    )
    
    parser.add_argument(
        "--from-file",
        type=str,
        help="Load predictions from existing file"
    )
    parser.add_argument(
        "--run-prediction",
        action="store_true",
        help="Run new predictions"
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=config.MULTI_PAIR_CONFIG.get("num_groups", 5),
        help=f"Number of test groups (default: {config.MULTI_PAIR_CONFIG.get('num_groups', 5)})"
    )
    parser.add_argument(
        "--pairs-per-group",
        type=int,
        default=config.MULTI_PAIR_CONFIG.get("pairs_per_group", 3),
        help=f"Number of pairs per group (default: {config.MULTI_PAIR_CONFIG.get('pairs_per_group', 3)})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation, only generate predictions"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    config.ensure_output_dir()
    
    if args.from_file:
        # Load predictions from file
        print(f"Loading predictions from file: {args.from_file}")
        with open(args.from_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        predictions_per_group = data["predictions_per_group"]
        groundtruths_per_group = data.get("groundtruths_per_group", [])
        model = None
        
    elif args.run_prediction:
        print("Running new multi-pair predictions...")
        
        # Check local model
        if not config.check_local_model():
            print("ERROR: Local model not found!")
            print(f"   Please check path: {config.LOCAL_MODEL_PATH}")
            return
        
        # Get configuration
        num_groups = args.num_groups
        pairs_per_group = args.pairs_per_group
        
        print(f"\nConfiguration:")
        print(f"  Number of test groups: {num_groups}")
        print(f"  Pairs per group: {pairs_per_group}")
        print(f"  Total test samples: {num_groups * pairs_per_group}")
        
        # Load data
        data_loader = BenchmarkDataLoader(config.DATA_PATH)
        total_samples = num_groups * pairs_per_group
        all_test_cases = data_loader.get_test_cases(total_samples)
        
        if len(all_test_cases) < total_samples:
            print(f"Warning: Dataset only has {len(all_test_cases)} samples, less than requested {total_samples}")
            total_samples = len(all_test_cases)
            num_groups = total_samples // pairs_per_group
            print(f"   Adjusted to: {num_groups} groups x {pairs_per_group} pairs")
        
        # Split into groups
        test_cases_per_group = []
        groundtruths_per_group = []
        
        for i in range(num_groups):
            start_idx = i * pairs_per_group
            end_idx = start_idx + pairs_per_group
            group_cases = all_test_cases[start_idx:end_idx]
            
            test_cases_per_group.append(group_cases)
            groundtruths_per_group.append([tc["groundtruth"] for tc in group_cases])
        
        # Initialize model
        print("\nLoading local model...")
        model = LocalModelInterface(**config.LOCAL_MODEL_CONFIG)
        
        # Run predictions
        predictions_per_group, raw_outputs = run_multi_pair_predictions(
            model,
            test_cases_per_group,
            temperature=args.temperature
        )
        
        # Save predictions
        timestamp = get_timestamp()
        pred_dir = os.path.join(config.OUTPUT_DIR, f"multi_pair_predictions_{timestamp}")
        os.makedirs(pred_dir, exist_ok=True)
        
        pred_data = {
            "metadata": {
                "timestamp": timestamp,
                "num_groups": num_groups,
                "pairs_per_group": pairs_per_group,
                "total_samples": num_groups * pairs_per_group,
                "temperature": args.temperature,
                "model_path": config.LOCAL_MODEL_PATH
            },
            "predictions_per_group": predictions_per_group,
            "groundtruths_per_group": groundtruths_per_group,
        }
        
        pred_path = os.path.join(pred_dir, "predictions.json")
        save_json(pred_data, pred_path)
        
        # Save raw outputs
        raw_output_path = os.path.join(pred_dir, "raw_outputs.txt")
        save_prediction_text(raw_outputs, raw_output_path)
        
        print(f"\nPredictions saved to: {pred_dir}")
        
    else:
        print("ERROR: Please specify --from-file or --run-prediction")
        parser.print_help()
        return
    
    # Skip evaluation if requested
    if args.skip_eval:
        print("\nSkipping evaluation phase")
        return
    
    # Run evaluation
    results = run_multi_pair_evaluation(
        predictions_per_group,
        groundtruths_per_group,
        model
    )
    
    if results is None:
        return
    
    # Save evaluation results
    timestamp = get_timestamp()
    output_dir = os.path.join(config.OUTPUT_DIR, f"multi_pair_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        "timestamp": timestamp,
        "num_groups": len(predictions_per_group),
        "model_path": config.LOCAL_MODEL_PATH,
        "llm_evaluator": config.LLM_EVALUATOR_CONFIG["model_name"],
    }
    
    save_data = {
        "metadata": metadata,
        "statistics": results["statistics"],
        "all_groups": results["all_groups"]
    }
    
    results_path = os.path.join(output_dir, "results.json")
    save_json(save_data, results_path)
    
    # Save failed cases
    failed_groups = [g for g in results["all_groups"] if not g["all_pass"]]
    if failed_groups:
        failed_path = os.path.join(output_dir, "failed_groups.json")
        save_json({
            "total_failed": len(failed_groups),
            "failed_groups": failed_groups
        }, failed_path)
    
    # Print summary
    print_summary(results, metadata)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - results.json")
    if failed_groups:
        print(f"  - failed_groups.json ({len(failed_groups)} failed groups)")


if __name__ == "__main__":
    main()
