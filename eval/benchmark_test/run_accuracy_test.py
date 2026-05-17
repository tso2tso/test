"""
Accuracy Test Script
Test model prediction accuracy using LLM binary judgment
Uses local model inference + multiagent_evaluator
"""
import os
import sys
import json
from typing import Dict, Any, List

# Get current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add project paths
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Use explicit import to avoid config conflict with root directory
import importlib.util
spec = importlib.util.spec_from_file_location("inference_config", os.path.join(SCRIPT_DIR, "config.py"))
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Use unified local model interface and utility functions
from local_model_interface import LocalModelInterface
from utils import save_json, get_timestamp, BenchmarkDataLoader, save_prediction_text

# Use unified multiagent_evaluator
sys.path.insert(0, os.path.join(PROJECT_ROOT, "eval"))
from evaluators import ThresholdCalibrationEvaluator


def run_accuracy_test(
    predictions: List[Dict[str, str]],
    groundtruths: List[Dict[str, str]],
    test_inputs: List[Dict[str, Any]] = None,
    model: LocalModelInterface = None
) -> Dict[str, Any]:
    """
    Run accuracy test using LLM binary judgment
    
    Args:
        predictions: List of model predictions
        groundtruths: List of ground truths
        test_inputs: Optional test inputs for reference
        model: LocalModelInterface for re-prediction
        
    Returns:
        Accuracy evaluation results
    """
    print("=" * 80)
    print("Starting Accuracy Test (LLM Binary Judgment)")
    print("=" * 80)
    
    # Check LLM evaluator configuration
    if not config.LLM_EVALUATOR_CONFIG.get("api_key"):
        print("\nERROR: LLM evaluator not configured!")
        print("Please configure LLM_EVALUATOR_CONFIG in config.py")
        return None
    
    # Initialize evaluator
    print("\nInitializing LLM Evaluator...")
    print(f"  Model: {config.LLM_EVALUATOR_CONFIG['model_name']}")
    print(f"  API: {config.LLM_EVALUATOR_CONFIG['api_url']}")
    
    evaluator = ThresholdCalibrationEvaluator(
        semantic_model_name=config.get_semantic_model_path(),
        llm_api_url=config.LLM_EVALUATOR_CONFIG["api_url"],
        llm_api_key=config.LLM_EVALUATOR_CONFIG["api_key"],
        llm_model_name=config.LLM_EVALUATOR_CONFIG["model_name"],
        same_config=config.LLM_SAME_CONFIG
    )

    # Evaluate both fields
    print(f"\nEvaluating {len(predictions)} samples...")
    print("=" * 80)
    
    results = {
        "FaultDescription": [],
        "RepairMeasures": []
    }
    
    # Process each sample
    for i, (pred, gt) in enumerate(zip(predictions, groundtruths)):
        print(f"\nSample {i+1}/{len(predictions)}")
        
        # FaultDescription
        fault_pred = pred.get("FaultDescription", "")
        fault_gt = gt.get("FaultDescription", "")
        
        print(f"  Evaluating FaultDescription...")
        fault_judgment = evaluator.llm_binary_judgment("FaultDescription", fault_gt, fault_pred)

        # Retry mechanism
        for retry in range(3):
            if fault_judgment["judgment"] == "符合":
                break

            print(f"  FaultDescription mismatch, retry {retry+1}/3...")
            if model is not None:
                new_pred = model.re_predict_faultdescription(pred)
                fault_pred = new_pred.get("FaultDescription", "")
                fault_judgment = evaluator.llm_binary_judgment("FaultDescription", fault_gt, fault_pred)
        
        results["FaultDescription"].append({
            "ecu_id": pred.get("ecu_id", ""),
            "dtc": pred.get("dtc", ""),
            "trigger": pred.get("trigger", ""),
            "timecondition": pred.get("timecondition", ""),
            "prediction": fault_pred,
            "groundtruth": fault_gt,
            "judgment": fault_judgment["judgment"],
            "pass": fault_judgment["judgment"] == "符合"
        })
        
        print(f"    Result: {fault_judgment['judgment']}")

        # RepairMeasures
        repair_pred = pred.get("RepairMeasures", "")
        repair_gt = gt.get("RepairMeasures", "")
        
        print(f"  Evaluating RepairMeasures...")
        repair_judgment = evaluator.llm_binary_judgment("RepairMeasures", repair_gt, repair_pred)

        # Retry mechanism
        for retry in range(3):
            if repair_judgment["judgment"] == "符合":
                break

            print(f"  RepairMeasures mismatch, retry {retry+1}/3...")
            if model is not None:
                new_pred = model.re_predict_repairmeasures(pred, fault_judgment["judgment"])
                repair_pred = new_pred.get("RepairMeasures", "")
                repair_judgment = evaluator.llm_binary_judgment("RepairMeasures", repair_gt, repair_pred)
        
        results["RepairMeasures"].append({
            "ecu_id": pred.get("ecu_id", ""),
            "dtc": pred.get("dtc", ""),
            "trigger": pred.get("trigger", ""),
            "timecondition": pred.get("timecondition", ""),
            "prediction": repair_pred,
            "groundtruth": repair_gt,
            "judgment": repair_judgment["judgment"],
            "pass": repair_judgment["judgment"] == "符合"
        })
        
        print(f"    Result: {repair_judgment['judgment']}")
    
    # Calculate statistics
    summary = {}
    for field in ["FaultDescription", "RepairMeasures"]:
        field_results = results[field]
        total = len(field_results)
        passed = sum(1 for r in field_results if r["pass"])
        
        summary[field] = {
            "total_samples": total,
            "passed": passed,
            "pass_rate": passed / total if total > 0 else 0
        }
    
    # Overall pass rate (both fields must pass)
    overall_passes = sum(
        1 for i in range(len(predictions))
        if results["FaultDescription"][i]["pass"] and results["RepairMeasures"][i]["pass"]
    )
    
    summary["overall"] = {
        "total_samples": len(predictions),
        "both_pass": overall_passes,
        "overall_pass_rate": overall_passes / len(predictions) if predictions else 0
    }
    
    return {
        "summary": summary,
        "detailed_results": results,
        "test_inputs": test_inputs
    }


def print_summary(results: Dict[str, Any], metadata: Dict[str, Any]):
    """Print accuracy test summary"""
    summary = results["summary"]
    
    print("\n" + "=" * 80)
    print("ACCURACY TEST SUMMARY")
    print("=" * 80)
    
    print(f"\nTest Configuration:")
    print(f"  Total Samples: {summary['overall']['total_samples']}")
    print(f"  Model: {metadata.get('model_path', 'Unknown')}")
    print(f"  LLM Evaluator: {metadata['llm_evaluator']}")
    
    for field in ["FaultDescription", "RepairMeasures"]:
        fs = summary[field]
        print(f"\n{field}:")
        print(f"  Passed: {fs['passed']}/{fs['total_samples']}")
        print(f"  Pass Rate: {fs['pass_rate']:.2%}")
    
    print(f"\n{'='*80}")
    print("OVERALL RESULTS (Both Fields Pass)")
    print(f"{'='*80}")
    ov = summary["overall"]
    print(f"  Both Pass: {ov['both_pass']}/{ov['total_samples']}")
    print(f"  Overall Pass Rate: {ov['overall_pass_rate']:.2%}")
    print("=" * 80 + "\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run accuracy test with LLM binary judgment (Local Model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run new predictions with local model
  python run_accuracy_test.py --run-prediction --sample-size 20
  
  # Load predictions from file
  python run_accuracy_test.py --from-file results/predictions.json
        """
    )
    
    parser.add_argument(
        "--from-file",
        type=str,
        help="Load predictions from previous results file"
    )
    parser.add_argument(
        "--run-prediction",
        action="store_true",
        help="Run new predictions using local model"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=config.TEST_SAMPLE_SIZE,
        help=f"Number of test cases (default: {config.TEST_SAMPLE_SIZE})"
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
    
    model = None
    raw_outputs = []
    
    # Load or generate predictions
    if args.from_file:
        print(f"Loading predictions from: {args.from_file}")
        with open(args.from_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        predictions = data.get("predictions", [])
        groundtruths = data.get("groundtruths", [])
        test_inputs = data.get("test_inputs", [])
        
    elif args.run_prediction:
        print("Running new predictions with local model...")
        
        # Check local model
        if not config.check_local_model():
            print("ERROR: Local model not found!")
            print(f"   Please check path: {config.LOCAL_MODEL_PATH}")
            return
        
        # Load data
        data_loader = BenchmarkDataLoader(config.DATA_PATH)
        test_cases = data_loader.get_test_cases(args.sample_size)
        
        print(f"\nLoaded {len(test_cases)} test cases")
        
        # Initialize local model
        print("\nLoading local model...")
        model = LocalModelInterface(**config.LOCAL_MODEL_CONFIG)
        
        # Run predictions
        predictions = []
        print(f"\nGenerating predictions for {len(test_cases)} samples...")
        
        for i, tc in enumerate(test_cases):
            print(f"  Predicting {i+1}/{len(test_cases)}")
            pred, raw_result = model.predict(
                ecu_id=tc["input"]["ECU_ID"],
                dtc=tc["input"]["DTC"],
                trigger=tc["input"]["Trigger"],
                timecondition=tc["input"]["TimeCondition"],
                temperature=args.temperature
            )
            
            pred["FaultDescription_gt"] = tc["groundtruth"]["FaultDescription"]
            pred["RepairMeasures_gt"] = tc["groundtruth"]["RepairMeasures"]
            
            predictions.append(pred)
            raw_outputs.append({
                "index": i + 1,
                "input": tc["input"],
                "raw_output": raw_result
            })
        
        groundtruths = [tc["groundtruth"] for tc in test_cases]
        test_inputs = [tc["input"] for tc in test_cases]
        
        # Save predictions
        timestamp = get_timestamp()
        pred_dir = os.path.join(config.OUTPUT_DIR, f"predictions_{timestamp}")
        os.makedirs(pred_dir, exist_ok=True)
        
        pred_data = {
            "metadata": {
                "timestamp": timestamp,
                "num_samples": len(predictions),
                "temperature": args.temperature,
                "model_path": config.LOCAL_MODEL_PATH
            },
            "predictions": predictions,
            "groundtruths": groundtruths,
            "test_inputs": test_inputs
        }
        
        pred_path = os.path.join(pred_dir, "predictions.json")
        save_json(pred_data, pred_path)
        
        # Save raw outputs
        raw_output_path = os.path.join(pred_dir, "raw_outputs.txt")
        save_prediction_text(raw_outputs, raw_output_path)
        
        print(f"\nPredictions saved to: {pred_dir}")
    
    else:
        print("ERROR: Please specify either --from-file or --run-prediction")
        parser.print_help()
        return
    
    # Skip evaluation if requested
    if args.skip_eval:
        print("\nSkipping evaluation phase")
        return
    
    # Run accuracy test
    results = run_accuracy_test(predictions, groundtruths, test_inputs, model)
    
    if results is None:
        return
    
    # Save results
    timestamp = get_timestamp()
    output_dir = os.path.join(config.OUTPUT_DIR, f"accuracy_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        "timestamp": timestamp,
        "num_samples": len(predictions),
        "model_path": config.LOCAL_MODEL_PATH,
        "llm_evaluator": config.LLM_EVALUATOR_CONFIG["model_name"],
        "semantic_model": config.get_semantic_model_path()
    }
    
    save_data = {
        "metadata": metadata,
        "summary": results["summary"],
        "detailed_results": results["detailed_results"]
    }
    
    save_path = os.path.join(output_dir, "accuracy_results.json")
    save_json(save_data, save_path)
    
    # Save failed cases
    failed_cases = []
    for i in range(len(predictions)):
        fault_result = results["detailed_results"]["FaultDescription"][i]
        repair_result = results["detailed_results"]["RepairMeasures"][i]
        
        if not (fault_result["pass"] and repair_result["pass"]):
            failed_cases.append({
                "index": i,
                "input": results["test_inputs"][i] if results["test_inputs"] else None,
                "FaultDescription": fault_result,
                "RepairMeasures": repair_result
            })
    
    if failed_cases:
        failed_path = os.path.join(output_dir, "failed_cases.json")
        save_json({"total_failed": len(failed_cases), "cases": failed_cases}, failed_path)
    
    # Print summary
    print_summary(results, metadata)
    
    print(f"Results saved to: {output_dir}")
    print(f"  - accuracy_results.json")
    if failed_cases:
        print(f"  - failed_cases.json ({len(failed_cases)} failed cases)")


if __name__ == "__main__":
    main()
