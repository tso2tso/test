"""
Multi-Turn Accuracy Test Script
Test model prediction accuracy in multi-turn conversation scenarios
Each session contains multiple turns, maintaining conversation history context
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


def run_multi_turn_predictions(
    model: LocalModelInterface,
    test_cases_per_session: List[List[Dict[str, Any]]],
    temperature: float = 0.7
) -> tuple:
    """
    Run multi-turn conversation predictions
    
    Args:
        model: Model interface
        test_cases_per_session: Test cases grouped by session
        temperature: Sampling temperature
        
    Returns:
        (predictions list, raw output records)
    """
    all_predictions = []
    all_raw_outputs = []
    
    print(f"\n{'='*80}")
    print(f"Starting Multi-Turn Predictions")
    print(f"{'='*80}")
    print(f"Number of sessions: {len(test_cases_per_session)}")
    
    for session_idx, test_cases in enumerate(test_cases_per_session):
        print(f"\n--- Session {session_idx + 1}/{len(test_cases_per_session)} ---")
        print(f"Turns in this session: {len(test_cases)}")
        
        # Reset conversation history
        model.reset_conversation()
        
        session_predictions = []
        session_raw_outputs = []
        
        for turn_idx, test_case in enumerate(test_cases):
            print(f"  Turn {turn_idx + 1}...")
            input_data = test_case["input"]
            groundtruth = test_case["groundtruth"]
            
            # Use prediction with history (multi-turn conversation)
            prediction = model.predict_with_history(
                ecu_id=input_data["ECU_ID"],
                dtc=input_data["DTC"],
                trigger=input_data["Trigger"],
                timecondition=input_data["TimeCondition"],
                temperature=temperature
            )
            
            prediction["FaultDescription_gt"] = groundtruth["FaultDescription"]
            prediction["RepairMeasures_gt"] = groundtruth["RepairMeasures"]
            
            session_predictions.append(prediction)
            session_raw_outputs.append({
                "index": f"S{session_idx + 1}-T{turn_idx + 1}",
                "input": input_data,
                "raw_output": {
                    "FaultDescription": prediction["FaultDescription"],
                    "RepairMeasures": prediction["RepairMeasures"],
                    "conversation_turns": len(model.conversation_history) // 2
                }
            })
            
            print(f"    Done (history turns: {len(model.conversation_history) // 2})")
        
        all_predictions.append(session_predictions)
        all_raw_outputs.extend(session_raw_outputs)
    
    return all_predictions, all_raw_outputs


def run_multi_turn_evaluation(
    predictions_per_session: List[List[Dict[str, str]]],
    groundtruths_per_session: List[List[Dict[str, str]]],
    model: LocalModelInterface = None
) -> Dict[str, Any]:
    """
    Run multi-turn conversation evaluation
    
    Args:
        predictions_per_session: Predictions grouped by session
        groundtruths_per_session: Ground truths grouped by session
        model: Model interface (for retry)
        
    Returns:
        Evaluation results
    """
    print("\n" + "=" * 80)
    print("Starting Multi-Turn Accuracy Evaluation")
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
    
    # Statistics variables
    all_session_results = []
    max_turns = max(len(session) for session in predictions_per_session)
    
    # Per-turn statistics
    turn_statistics = {i: {
        "FaultDescription": {"pass": 0, "fail": 0, "scores": []},
        "RepairMeasures": {"pass": 0, "fail": 0, "scores": []},
        "overall": {"pass": 0, "fail": 0}
    } for i in range(max_turns)}
    
    for session_idx, (preds, gts) in enumerate(zip(predictions_per_session, groundtruths_per_session)):
        print(f"\n{'='*60}")
        print(f"Evaluating Session {session_idx + 1}/{len(predictions_per_session)}")
        print(f"{'='*60}")
        
        session_result = {
            "session_index": session_idx,
            "num_turns": len(preds),
            "turn_results": [],
            "session_pass": True
        }
        
        for turn_idx, (pred, gt) in enumerate(zip(preds, gts)):
            print(f"\n  Turn {turn_idx + 1}:")
            
            turn_result = {
                "turn_index": turn_idx,
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
            
            fault_pass = fault_judgment["judgment"] == "符合"
            turn_result["FaultDescription"] = {
                "prediction": fault_pred,
                "groundtruth": fault_gt,
                "judgment": fault_judgment["judgment"],
                "score": fault_judgment.get("score", 0),
                "pass": fault_pass
            }
            
            # Update turn statistics
            if fault_pass:
                turn_statistics[turn_idx]["FaultDescription"]["pass"] += 1
            else:
                turn_statistics[turn_idx]["FaultDescription"]["fail"] += 1
            turn_statistics[turn_idx]["FaultDescription"]["scores"].append(fault_judgment.get("score", 0))
            
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
            
            repair_pass = repair_judgment["judgment"] == "符合"
            turn_result["RepairMeasures"] = {
                "prediction": repair_pred,
                "groundtruth": repair_gt,
                "judgment": repair_judgment["judgment"],
                "score": repair_judgment.get("score", 0),
                "pass": repair_pass
            }
            
            # Update turn statistics
            if repair_pass:
                turn_statistics[turn_idx]["RepairMeasures"]["pass"] += 1
            else:
                turn_statistics[turn_idx]["RepairMeasures"]["fail"] += 1
            turn_statistics[turn_idx]["RepairMeasures"]["scores"].append(repair_judgment.get("score", 0))
            
            print(f"      Result: {repair_judgment['judgment']} (Score: {repair_judgment.get('score', 0)})")
            
            # Check if this turn passed
            turn_pass = fault_pass and repair_pass
            turn_result["pass"] = turn_pass
            
            if turn_pass:
                turn_statistics[turn_idx]["overall"]["pass"] += 1
            else:
                turn_statistics[turn_idx]["overall"]["fail"] += 1
                session_result["session_pass"] = False
            
            session_result["turn_results"].append(turn_result)
        
        # Session result
        if session_result["session_pass"]:
            print(f"\n  Session {session_idx + 1} ALL TURNS PASSED")
        else:
            print(f"\n  Session {session_idx + 1} HAS FAILED TURNS")
        
        all_session_results.append(session_result)
    
    # Calculate statistics
    # Calculate turn-level pass rate (all passed turns / total turns)
    total_turns = 0
    total_passed_turns = 0
    total_fault_passed = 0
    total_repair_passed = 0
    
    for turn_idx in range(max_turns):
        ts = turn_statistics[turn_idx]
        turn_total = ts["FaultDescription"]["pass"] + ts["FaultDescription"]["fail"]
        total_turns += turn_total
        total_passed_turns += ts["overall"]["pass"]
        total_fault_passed += ts["FaultDescription"]["pass"]
        total_repair_passed += ts["RepairMeasures"]["pass"]
    
    statistics = {
        "total_sessions": len(predictions_per_session),
        "max_turns": max_turns,
        "turn_statistics": [],
        "session_statistics": {
            "all_pass": sum(1 for s in all_session_results if s["session_pass"]),
            "any_fail": sum(1 for s in all_session_results if not s["session_pass"])
        },
        # Turn-level statistics
        "turn_level_statistics": {
            "total_turns": total_turns,
            "passed_turns": total_passed_turns,
            "turn_pass_rate": total_passed_turns / total_turns if total_turns > 0 else 0,
            "fault_passed": total_fault_passed,
            "fault_pass_rate": total_fault_passed / total_turns if total_turns > 0 else 0,
            "repair_passed": total_repair_passed,
            "repair_pass_rate": total_repair_passed / total_turns if total_turns > 0 else 0,
        }
    }
    
    for turn_idx in range(max_turns):
        ts = turn_statistics[turn_idx]
        total = ts["FaultDescription"]["pass"] + ts["FaultDescription"]["fail"]
        
        if total > 0:
            fault_scores = ts["FaultDescription"]["scores"]
            repair_scores = ts["RepairMeasures"]["scores"]
            
            statistics["turn_statistics"].append({
                "turn_index": turn_idx,
                "num_samples": total,
                "FaultDescription": {
                    "pass_rate": ts["FaultDescription"]["pass"] / total,
                    "avg_score": sum(fault_scores) / len(fault_scores) if fault_scores else 0
                },
                "RepairMeasures": {
                    "pass_rate": ts["RepairMeasures"]["pass"] / total,
                    "avg_score": sum(repair_scores) / len(repair_scores) if repair_scores else 0
                },
                "overall": {
                    "pass_rate": ts["overall"]["pass"] / total
                }
            })
    
    return {
        "statistics": statistics,
        "all_sessions": all_session_results
    }


def print_summary(results: Dict[str, Any], metadata: Dict[str, Any]):
    """Print test summary"""
    stats = results["statistics"]
    
    print("\n" + "=" * 80)
    print("Multi-Turn Accuracy Test Summary")
    print("=" * 80)
    
    print(f"\nTest Configuration:")
    print(f"  Total Sessions: {stats['total_sessions']}")
    print(f"  Max Turns: {stats['max_turns']}")
    print(f"  Model: {metadata.get('model_path', 'Unknown')}")
    print(f"  Evaluator: {metadata.get('llm_evaluator', 'Unknown')}")
    
    # Turn-level pass rate (core metric)
    tls = stats.get("turn_level_statistics", {})
    if tls:
        print(f"\n{'='*40}")
        print(f"Turn-Level Pass Rate")
        print(f"{'='*40}")
        print(f"  Total Turns: {tls['total_turns']}")
        print(f"  Passed Turns: {tls['passed_turns']}")
        print(f"  Turn Pass Rate: {tls['turn_pass_rate']:.2%} ({tls['passed_turns']}/{tls['total_turns']})")
        print(f"  ")
        print(f"  FaultDescription Passed: {tls['fault_passed']}/{tls['total_turns']} ({tls['fault_pass_rate']:.2%})")
        print(f"  RepairMeasures Passed: {tls['repair_passed']}/{tls['total_turns']} ({tls['repair_pass_rate']:.2%})")
    
    print(f"\nSession-Level Results (all turns must pass for session to pass):")
    print(f"  All Pass Sessions: {stats['session_statistics']['all_pass']}")
    print(f"  Has Failed Sessions: {stats['session_statistics']['any_fail']}")
    print(f"  Session Pass Rate: {stats['session_statistics']['all_pass'] / stats['total_sessions']:.2%}")
    
    print(f"\nPer-Turn Accuracy Statistics:")
    print(f"{'Turn':<6} {'Samples':<8} {'FaultDesc Pass Rate':<20} {'Repair Pass Rate':<20} {'Overall':<15}")
    print("-" * 80)
    
    for ts in stats["turn_statistics"]:
        turn_num = ts["turn_index"] + 1
        num_samples = ts["num_samples"]
        fault_rate = ts["FaultDescription"]["pass_rate"]
        repair_rate = ts["RepairMeasures"]["pass_rate"]
        overall_rate = ts["overall"]["pass_rate"]
        
        print(f"{turn_num:<6} {num_samples:<8} {fault_rate:>6.2%} (avg: {ts['FaultDescription']['avg_score']:.1f})   "
              f"{repair_rate:>6.2%} (avg: {ts['RepairMeasures']['avg_score']:.1f})   {overall_rate:>6.2%}")
    
    print("\n" + "=" * 80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run multi-turn accuracy test (using local fine-tuned model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run new multi-turn prediction test
  python run_multi_turn_accuracy_test.py --run-prediction --num-sessions 5 --turns-per-session 3
  
  # Only generate predictions, skip evaluation
  python run_multi_turn_accuracy_test.py --run-prediction --skip-eval
  
  # Use existing prediction results for evaluation
  python run_multi_turn_accuracy_test.py --from-file results/xxx/predictions.json
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
        "--num-sessions",
        type=int,
        default=config.MULTI_TURN_CONFIG.get("num_sessions", 5),
        help=f"Number of sessions (default: {config.MULTI_TURN_CONFIG.get('num_sessions', 5)})"
    )
    parser.add_argument(
        "--turns-per-session",
        type=int,
        default=config.MULTI_TURN_CONFIG.get("turns_per_session", 3),
        help=f"Turns per session (default: {config.MULTI_TURN_CONFIG.get('turns_per_session', 3)})"
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
        
        predictions_per_session = data["predictions_per_session"]
        groundtruths_per_session = data.get("groundtruths_per_session", [])
        model = None
        
    elif args.run_prediction:
        print("Running new multi-turn predictions...")
        
        # Check local model
        if not config.check_local_model():
            print("ERROR: Local model not found!")
            print(f"   Please check path: {config.LOCAL_MODEL_PATH}")
            return
        
        # Get configuration
        num_sessions = args.num_sessions
        turns_per_session = args.turns_per_session
        
        print(f"\nConfiguration:")
        print(f"  Number of sessions: {num_sessions}")
        print(f"  Turns per session: {turns_per_session}")
        print(f"  Total test samples: {num_sessions * turns_per_session}")
        
        # Load data
        data_loader = BenchmarkDataLoader(config.DATA_PATH)
        total_samples = num_sessions * turns_per_session
        all_test_cases = data_loader.get_test_cases(total_samples)
        
        if len(all_test_cases) < total_samples:
            print(f"Warning: Dataset only has {len(all_test_cases)} samples, less than requested {total_samples}")
            total_samples = len(all_test_cases)
            num_sessions = total_samples // turns_per_session
            print(f"   Adjusted to: {num_sessions} sessions x {turns_per_session} turns")
        
        # Split into sessions
        test_cases_per_session = []
        groundtruths_per_session = []
        
        for i in range(num_sessions):
            start_idx = i * turns_per_session
            end_idx = start_idx + turns_per_session
            session_cases = all_test_cases[start_idx:end_idx]
            
            test_cases_per_session.append(session_cases)
            groundtruths_per_session.append([tc["groundtruth"] for tc in session_cases])
        
        # Initialize model
        print("\nLoading local model...")
        model = LocalModelInterface(**config.LOCAL_MODEL_CONFIG)
        
        # Run predictions
        predictions_per_session, raw_outputs = run_multi_turn_predictions(
            model,
            test_cases_per_session,
            temperature=args.temperature
        )
        
        # Save predictions
        timestamp = get_timestamp()
        pred_dir = os.path.join(config.OUTPUT_DIR, f"multi_turn_predictions_{timestamp}")
        os.makedirs(pred_dir, exist_ok=True)
        
        pred_data = {
            "metadata": {
                "timestamp": timestamp,
                "num_sessions": num_sessions,
                "turns_per_session": turns_per_session,
                "total_samples": num_sessions * turns_per_session,
                "temperature": args.temperature,
                "model_path": config.LOCAL_MODEL_PATH
            },
            "predictions_per_session": predictions_per_session,
            "groundtruths_per_session": groundtruths_per_session,
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
    results = run_multi_turn_evaluation(
        predictions_per_session,
        groundtruths_per_session,
        model
    )
    
    if results is None:
        return
    
    # Save evaluation results
    timestamp = get_timestamp()
    output_dir = os.path.join(config.OUTPUT_DIR, f"multi_turn_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        "timestamp": timestamp,
        "num_sessions": len(predictions_per_session),
        "model_path": config.LOCAL_MODEL_PATH,
        "llm_evaluator": config.LLM_EVALUATOR_CONFIG["model_name"],
    }
    
    save_data = {
        "metadata": metadata,
        "statistics": results["statistics"],
        "all_sessions": results["all_sessions"]
    }
    
    results_path = os.path.join(output_dir, "results.json")
    save_json(save_data, results_path)
    
    # Save failed cases (by turn)
    for turn_idx in range(results["statistics"]["max_turns"]):
        failed_turns = []
        
        for session in results["all_sessions"]:
            if turn_idx < len(session["turn_results"]):
                turn_result = session["turn_results"][turn_idx]
                if not turn_result["pass"]:
                    failed_turns.append({
                        "session_index": session["session_index"],
                        "turn_index": turn_idx,
                        "FaultDescription": turn_result["FaultDescription"],
                        "RepairMeasures": turn_result["RepairMeasures"]
                    })
        
        if failed_turns:
            failed_path = os.path.join(output_dir, f"failed_turn_{turn_idx + 1}.json")
            save_json({
                "turn": turn_idx + 1,
                "total_failed": len(failed_turns),
                "cases": failed_turns
            }, failed_path)
    
    # Print summary
    print_summary(results, metadata)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - results.json")
    
    # Count failed files
    failed_files = [f for f in os.listdir(output_dir) if f.startswith("failed_turn_")]
    if failed_files:
        print(f"  - Failed case files: {len(failed_files)}")


if __name__ == "__main__":
    main()
