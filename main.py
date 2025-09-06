import os
import sys
import torch
import traceback
from rich import print as rprint
import matplotlib.pyplot as plt
import datetime
import io
import json
import time
import copy
import re
import random

try:
    import pygraphviz
    PYGRAPHVIZ_AVAILABLE = True
except ImportError:
    PYGRAPHVIZ_AVAILABLE = False


from config import (MODEL_CONFIGS, TASK_CONFIGS, SYNTHE_TASK_CONFIGS, USE_SYNTHETIC_DATASET as GLOBAL_USE_SYNTHETIC_DATASET, SYNTHETIC_DATA_FILE_PATH,
                    EAP_CONFIG, ANALYSIS_CONFIG, parse_args, EDGE_OPTIMIZATION_CONFIG,
                    timestamp as config_timestamp, OUTPUT_DIR as CONFIG_DEFAULT_OUTPUT_DIR, EPOCHS,
                    SELECTED_TASK_INDICES as CONFIG_SELECTED_TASK_INDICES, CO_XSUB_MODE, DEVICE_CONFIG)
from model_loader import load_model
from data_utils import prepare_task_data
from analysis import get_component_logits_local, analyze_layer_outputs, analyze_specific_components, analyze_attention_for_subjects
from eap_utils import (run_eap_analysis, visualize_graph, evaluate_circuit_performance, 
                       get_graphviz_elements, compare_graph_elements, optimize_edge_count)
from eap.evaluate import get_circuit_logits
from co_xsub_identifier import identify_xsub_by_co, identify_ysub_ydom, safe_to_str_tokens, safe_to_tokens



class TeeLogger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
        self.buffer = io.StringIO()
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.buffer.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()
        
    def get_log_content(self):
        return self.buffer.getvalue()

def main():
    """
    Main function: Execute EAP knowledge circuit analysis process
    """
    co_success_count = 0
    co_total_count = 0
    
    args = parse_args()
    
    model_name = args.model
    selected_task_indices = args.tasks
    max_new_tokens = args.max_new_tokens
    
    model_name_for_run = args.model
    target_edge_count_for_run = args.target_edges
    output_dir_argument = args.output_dir
    
    os.makedirs("output_info", exist_ok=True)
    
    if output_dir_argument == CONFIG_DEFAULT_OUTPUT_DIR:
        safe_model_name_for_dir = model_name_for_run.replace('/', '_')
        base_run_output_dir = os.path.join("output_info", f"eap_{safe_model_name_for_dir}_{target_edge_count_for_run}_{config_timestamp}")
    else:
        custom_dir_name = os.path.basename(output_dir_argument)
        base_run_output_dir = os.path.join("output_info", custom_dir_name)
    
    current_use_synthetic = GLOBAL_USE_SYNTHETIC_DATASET
    
    os.makedirs(base_run_output_dir, exist_ok=True)
    main_log_dir = base_run_output_dir
    
    visualization_flag = args.visualization
    
    eap_method = args.method
    ig_steps = args.ig_steps
    target_edge_count = args.target_edges
    visualize_circuit_flag = args.visualize_circuit

    
    ANALYSIS_CONFIG["max_new_tokens"] = max_new_tokens
    ANALYSIS_CONFIG["analyze_reversed_circuit"] = args.analyze_reversed_circuit 
    
    optimize_edges_flag = args.optimize_edges
    optimization_goal = args.optimization_goal
    initial_edge_count = args.initial_edges if args.initial_edges is not None else EAP_CONFIG["target_edge_count"]
    step_size = args.step_size
    edge_count_range = EDGE_OPTIMIZATION_CONFIG["edge_count_range"]
    target_performance = args.target_performance
    
    if optimize_edges_flag:
        print(f"Will execute edge optimization: Target = {optimization_goal}")
        print(f"   Optimization method: {EDGE_OPTIMIZATION_CONFIG['optimization_method']}")
        print(f"   Detailed analysis: {EDGE_OPTIMIZATION_CONFIG['detailed_analysis']}")
        print(f"   Initial edge count: {initial_edge_count}")
        if EDGE_OPTIMIZATION_CONFIG['optimization_method'] == "golden_section":
            print(f"   Step size (golden section): {step_size if step_size is not None else 'Auto (0.1% of total edges)'}")
        elif EDGE_OPTIMIZATION_CONFIG['optimization_method'] == "uniform_interval":
            print(f"   Uniform step size: {EDGE_OPTIMIZATION_CONFIG['uniform_step_size']}")
        if edge_count_range:
            print(f"   Edge count range: [{edge_count_range[0]}, {edge_count_range[1]}]")
        else:
            print(f"   Edge count range: Auto")
        if optimization_goal == "target" and target_performance is not None:
            print(f"   Target performance value: {target_performance}")
        if EDGE_OPTIMIZATION_CONFIG['detailed_analysis']:
            print(f"   Detailed analysis includes: performance, attention metrics, high attention heads count")

    if model_name not in MODEL_CONFIGS:
        print(f"Error: Invalid model name '{model_name}'. Available options: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    model_config = MODEL_CONFIGS[model_name]
    current_use_synthetic = GLOBAL_USE_SYNTHETIC_DATASET

    all_epochs_attention_metrics = []
    all_epochs_high_attention_metrics = []
    
    edge_optimization_results = {}

   
    for current_epoch in EPOCHS:
        print(f"\n{'*' * 100}")
        print(f"Starting to process EPOCH: {current_epoch}")
        print(f"{'-' * 100}")

        skip_current_epoch = False

        current_epoch_output_dir = os.path.join(base_run_output_dir, f"epoch_{current_epoch}")
        os.makedirs(current_epoch_output_dir, exist_ok=True)
        
        try:
            model_config_for_current_epoch = copy.deepcopy(MODEL_CONFIGS[model_name])

            if "path_template" in model_config_for_current_epoch:
                model_config_for_current_epoch["path"] = model_config_for_current_epoch["path_template"].format(epoch_num=current_epoch)
                print(f"Model path for Epoch {current_epoch}: {model_config_for_current_epoch['path']}")
            elif "path" not in model_config_for_current_epoch:
                 print(f"Error: Model '{model_name}' configuration lacks 'path' or 'path_template'. Skipping epoch {current_epoch}.")
                 raise ValueError(f"Configuration error for model {model_name}: Missing 'path' or 'path_template'.")

            model, global_device = load_model(model_name, model_config_for_current_epoch, DEVICE_CONFIG)
        except Exception as e:
            print(f"Model loading failed (Epoch {current_epoch}): {e}")
            traceback.print_exc()
            skip_current_epoch = True

        if skip_current_epoch:
            print(f"Due to model loading failure, skipping subsequent analysis for Epoch {current_epoch}.")
            continue

        
        active_task_configs_source = SYNTHE_TASK_CONFIGS if current_use_synthetic else TASK_CONFIGS
        selected_tasks = []
        if not active_task_configs_source:
            print(f"Error: Selected configuration list ({'SYNTHE_TASK_CONFIGS' if current_use_synthetic else 'TASK_CONFIGS'}) is empty. Cannot proceed.")
            sys.exit(1)
        
        for i in selected_task_indices:
            if 0 <= i < len(active_task_configs_source):
                task_conf = active_task_configs_source[i]
                if current_use_synthetic and 'name' not in task_conf:
                    task_conf['name'] = f"SyntheticTask{i}"
                selected_tasks.append(task_conf)
            else:
                print(f"Warning: Task index {i} is invalid for current dataset selection, skipping.")
        
        if not selected_tasks:
            print("Error: No valid tasks selected! Please check selected task indices and configurations.")
            sys.exit(1)
        
        
        eap_config = {
            "method": eap_method,
            "ig_steps": ig_steps,
            "target_edge_count": target_edge_count,
            "visualize_graph": visualize_circuit_flag
        }
        
        if current_use_synthetic:
            mode = 2
            mode_name = "synthetic_clean_vs_corrupt"
        else:
            mode = 2
            mode_name = "clean_vs_corrupt"
        
        main_py_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        main_log_dir = base_run_output_dir
        os.makedirs(main_log_dir, exist_ok=True)
        main_log_filename = f"analysis_run_{main_py_timestamp}.log"
        main_log_path = os.path.join(main_log_dir, main_log_filename)
        original_stdout = sys.stdout
        sys.stdout = TeeLogger(main_log_path)
        print(f"=== Knowledge Circuit Analysis Run Log ===")
        print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {model_name}")
        print(f"Tasks: {[task.get('name', f'UnnamedTask') for task in selected_tasks]}")
        print(f"Using synthetic dataset: {'Yes' if current_use_synthetic else 'No'}")
        if current_use_synthetic:
            print(f"Synthetic dataset file: {SYNTHETIC_DATA_FILE_PATH}")
        print(f"Mode: {mode}:{mode_name}")
        print(f"EAP method: {eap_method}")
        print(f"IG steps: {ig_steps}")
        print(f"Target retained edge count: {target_edge_count}")
        print(f"Analysis max new token count: {max_new_tokens}")
        if optimize_edges_flag:
            print(f"Edge optimization enabled: Yes")
            print(f"   Optimization goal: {optimization_goal}")
            print(f"   Optimization method: {EDGE_OPTIMIZATION_CONFIG['optimization_method']}")
            print(f"   Detailed analysis: {EDGE_OPTIMIZATION_CONFIG['detailed_analysis']}")
            if EDGE_OPTIMIZATION_CONFIG['optimization_method'] == "golden_section":
                print(f"   Maximum iterations: {EDGE_OPTIMIZATION_CONFIG['max_iterations']} times")
            elif EDGE_OPTIMIZATION_CONFIG['optimization_method'] == "uniform_interval":
                print(f"   Uniform step size: {EDGE_OPTIMIZATION_CONFIG['uniform_step_size']}")
            if optimization_goal == "target" and target_performance is not None:
                print(f"   Target performance value: {target_performance}")
            if EDGE_OPTIMIZATION_CONFIG['detailed_analysis']:
                print(f"   Detailed analysis includes: performance, attention metrics, high attention heads count")
        else:
            print(f"Edge optimization enabled: No")
        print(f"=== Configuration Information End ===\n")
        # Run mode=2 only
        print(f"\n{'#'*100}")
        print(f"Starting to run mode {mode}: {mode_name}")
        print(f"{'#'*100}")
        for task_idx, task_config in enumerate(selected_tasks):
            task_name_base = task_config.get("name", f"Task{task_idx}")
            print(f"\n{'='*80}")
            print(f"Starting to process task {task_idx+1}/{len(selected_tasks)}: {task_name_base}")
            print(f"{'='*80}")
            try:
                clean_tokens_orig, corrupted_tokens_orig, label_tensor_orig, \
                clean_prompt_text_orig, corrupted_prompt_text_orig = prepare_task_data(
                    model, 
                    task_config, 
                    mode=mode, # 只传2
                    use_synthetic_dataset=current_use_synthetic,
                    synthetic_data_file_path=SYNTHETIC_DATA_FILE_PATH if current_use_synthetic else None
                )
                # --- Determine Text Inputs based on Mode --- (Get the actual prompts used)
                if current_use_synthetic:
                    text1_for_run = clean_prompt_text_orig
                    text2_for_run = corrupted_prompt_text_orig
                else:
                    clean_prompt_base = task_config["prompt_template"].format(task_config["clean_subject"])
                    corrupted_prompt_base = task_config["prompt_template"].format(task_config["corrupted_subject"])
                    text1_for_run = clean_prompt_base
                    text2_for_run = corrupted_prompt_base
                run_suffix = ""
                prompt_for_analysis = text1_for_run
                current_clean_text = text1_for_run
                current_corrupted_text = text2_for_run
                label_tensor_current_run = label_tensor_orig
                task_name = f"{task_name_base}{run_suffix}"
                print(f"\n--- Analysis run: Epoch={current_epoch}, Task='{task_name}', Mode={mode_name} --- ")
                print(f"Current Clean Prompt: '{prompt_for_analysis}'")
                task_output_dir = os.path.join(
                    current_epoch_output_dir,
                    f"mode{mode}_{mode_name}",
                    task_name
                )
                task_log_filename = f"analysis_{task_name}_{mode_name}_{main_py_timestamp}.log"
                task_log_path = os.path.join(task_output_dir, task_log_filename)
                os.makedirs(os.path.dirname(task_log_path), exist_ok=True)
                if isinstance(sys.stdout, TeeLogger):
                    with open(task_log_path, 'w', encoding='utf-8') as task_log_file:
                        task_log_file.write(f"=== Knowledge Circuit Analysis Task Log ===\n")
                        task_log_file.write(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        task_log_file.write(f"Model: {model_name}\n")
                        task_log_file.write(f"Task: {task_name}\n")
                        task_log_file.write(f"Using synthetic dataset: {'Yes' if current_use_synthetic else 'No'}\n")
                        task_log_file.write(f"Mode: {mode}:{mode_name}\n")
                        task_log_file.write(f"EAP method: {eap_method}\n")
                        task_log_file.write(f"IG steps: {ig_steps}\n")
                        task_log_file.write(f"Target retained edge count: {target_edge_count}\n")
                        task_log_file.write(f"Analysis max new token count: {max_new_tokens}\n")
                        task_log_file.write(f"=== Configuration Information End ===\n\n")
                task_data_tensors = (clean_tokens_orig, corrupted_tokens_orig, label_tensor_orig)
                task_data_text = (text1_for_run, text2_for_run, label_tensor_orig)
                
                do_eap = True
                if hasattr(args, 'use_co_xsub') and args.use_co_xsub:
                    tokenizer = model.tokenizer
                    print("\n Automatically identifying X_sub, Y_sub, Y_dom:")
                    xsub_token, xsub_idx = identify_xsub_by_co(
                        prompt_for_analysis, model, tokenizer,
                        method=args.co_xsub_method if hasattr(args, 'co_xsub_method') else 'phantomcircuit',
                        top_k=10, verbose=True
                    )
                    print(f" Identified X_sub: '{xsub_token}' (position: {xsub_idx})")
                    ysub_token, ydom_token, detail = identify_ysub_ydom(
                        prompt_for_analysis, model, tokenizer,
                        correct_answer=task_config.get('correct_answer'),
                        incorrect_answer=task_config.get('incorrect_answer'),
                        top_k=10, verbose=True
                    )
                    print(f" Identified Y_sub: '{ysub_token}', Y_dom: '{ydom_token}'")
                    print(f" Detailed information: {detail}")
                    if hasattr(args, 'co_xsub_eap') and not args.co_xsub_eap:
                        do_eap = False
                
                if do_eap:
                    try:
                        actual_target_edge_count = target_edge_count
                        if optimize_edges_flag:
                            print(f"\n--- Starting edge optimization (Epoch={current_epoch}, Task='{task_name}') ---")
                            print(f"Optimization goal: {optimization_goal}")
                            optimization_output_dir = os.path.join(task_output_dir, "edge_optimization")
                            os.makedirs(optimization_output_dir, exist_ok=True)
                            try:
                                optimization_eap_config = eap_config.copy()
                                optimization_eap_config["target_edge_count"] = initial_edge_count
                                best_edge_count, best_performance, optimization_results = optimize_edge_count(
                                    model,
                                    task_data_tensors,
                                    task_data_text,
                                    optimization_eap_config,
                                    optimization_output_dir,
                                    f"{task_name}_optimization",
                                    global_device,
                                    start_edge_count=initial_edge_count,
                                    step_size=step_size,
                                    edge_count_range=edge_count_range,
                                    target_performance=target_performance,
                                    optimization_goal=optimization_goal,
                                    optimization_method=EDGE_OPTIMIZATION_CONFIG["optimization_method"],
                                    uniform_step_size=EDGE_OPTIMIZATION_CONFIG["uniform_step_size"],
                                    max_iterations=EDGE_OPTIMIZATION_CONFIG["max_iterations"],
                                    detailed_analysis=EDGE_OPTIMIZATION_CONFIG["detailed_analysis"],
                                    task_config=task_config
                                )
                                if best_edge_count is not None:
                                    print(f"\nEdge optimization completed: Found best edge count = {best_edge_count}")
                                    actual_target_edge_count = best_edge_count
                                    optimization_key = f"epoch{current_epoch}_task{task_name}_mode{mode}"
                                    edge_optimization_results[optimization_key] = {
                                        "epoch": current_epoch,
                                        "task_name": task_name,
                                        "mode": mode,
                                        "best_edge_count": best_edge_count,
                                        "best_performance": best_performance,
                                        "optimization_results": [(edge, perf, imp) for edge, perf, imp in optimization_results]
                                    }
                                else:
                                    print(f"\nEdge optimization failed, default edge count {target_edge_count} will be used")
                            except Exception as e:
                                print(f"Edge optimization process error: {e}")
                                traceback.print_exc()
                                print(f"Default edge count {target_edge_count} will be used")
                        current_eap_config = eap_config.copy()
                        current_eap_config["target_edge_count"] = actual_target_edge_count
                        print(f"\nUsing edge count: {actual_target_edge_count}" + (f" (optimized)" if optimize_edges_flag and actual_target_edge_count != target_edge_count else ""))
                        g = None
                        circuit_logits = None
                        try:
                            g, circuit_logits = run_eap_analysis(
                                model,
                                task_data_tensors,
                                task_data_text,
                                current_eap_config,
                                task_output_dir,
                                task_name,
                                global_device,
                                nodes_to_remove_from_circuit=ANALYSIS_CONFIG["nodes_to_remove_from_circuit"]
                            )
                        except Exception as e:
                            print(f"EAP analysis failed ({task_name}): {e}")
                            traceback.print_exc()
                            g = None
                        if isinstance(sys.stdout, TeeLogger):
                            logger = sys.stdout
                            with open(task_log_path, 'a', encoding='utf-8') as task_log_file:
                                task_log_file.write(logger.buffer.getvalue())
                            logger.buffer = io.StringIO()
                        # The subsequent EAP-related analysis process remains unchanged
                        gz = None
                        current_nodes_set = set()
                        current_edges_set = set()
                        if PYGRAPHVIZ_AVAILABLE and g is not None:
                            try:
                                gz = g.to_graphviz()
                                current_nodes_set, current_edges_set = get_graphviz_elements(gz)
                            except Exception as e_gz:
                                print(f"Failed to generate Graphviz object or extract elements ({task_name}): {e_gz}")
                                traceback.print_exc()
                                gz = None
                        elif not PYGRAPHVIZ_AVAILABLE:
                            print("Skipping Graphviz generation and comparison: pygraphviz not available.")
                        original_nodes_set = current_nodes_set
                        original_edges_set = current_edges_set
                        if visualize_circuit_flag and PYGRAPHVIZ_AVAILABLE and gz is not None:
                            visualize_graph(
                                g,
                                task_output_dir,
                                target_edge_count,
                                ig_steps,
                                PYGRAPHVIZ_AVAILABLE
                            )
                        elif visualize_circuit_flag and not PYGRAPHVIZ_AVAILABLE:
                            print("Skipping circuit visualization: pygraphviz not available.")
                        elif visualize_circuit_flag and gz is None:
                            print(f"Skipping circuit visualization: Unable to generate Graphviz object for '{task_name}'.")
                        analyze_circuit = ANALYSIS_CONFIG["analyze_circuit"]
                        evaluate_performance = ANALYSIS_CONFIG["evaluate_performance"]
                        top_k_eval = ANALYSIS_CONFIG["top_k_eval"]
                        baseline_performance = None
                        if evaluate_performance and g is not None and label_tensor_orig is not None:
                            try:
                                dataloader = [(clean_tokens_orig, corrupted_tokens_orig, label_tensor_orig)]
                                baseline, _ = evaluate_circuit_performance(model, g, dataloader)
                                baseline_performance = baseline
                            except Exception as e:
                                print(f"Failed to evaluate circuit performance ({task_name}): {e}")
                                traceback.print_exc()
                        if analyze_circuit and circuit_logits is not None and label_tensor_orig is not None:
                            print(f"\n--- Analyzing circuit Logits (Input: '{prompt_for_analysis}') ---")
                            try:
                                logits_analysis_results = get_component_logits_local(
                                    circuit_logits,
                                    model,
                                    answer_token=label_tensor_orig[0, 0].item(),
                                    top_k=top_k_eval,
                                    compare_with_full_model=True,
                                    input_tokens=clean_tokens_orig,
                                    max_new_tokens=ANALYSIS_CONFIG["max_new_tokens"]
                                )
                            except Exception as e:
                                print(f"Failed to analyze circuit logits ({task_name}): {e}")
                                traceback.print_exc()
                        attention_metrics_for_task = {}
                        if ANALYSIS_CONFIG["analyze_attention_patterns"] and g is not None and label_tensor_orig is not None:
                            print(f"\n--- Analyzing attention patterns (Epoch {current_epoch}, Task: {task_name}) ---")
                            if not ANALYSIS_CONFIG["save_attention_plots"]:
                                print("Note: Attention image saving disabled, only retaining metadata")
                            try:
                                is_synthetic = current_use_synthetic
                                attention_metrics_for_task = analyze_attention_for_subjects(
                                    model,
                                    g,
                                    clean_tokens_orig,
                                    corrupted_tokens_orig,
                                    task_config,
                                    task_output_dir,
                                    task_name,
                                    is_synthetic=is_synthetic,
                                    current_epoch=current_epoch,
                                    save_plots=ANALYSIS_CONFIG["save_attention_plots"]
                                )
                                attention_metrics_filename = os.path.join(task_output_dir, f"{task_name}_attention_metrics_epoch{current_epoch}.json")
                                with open(attention_metrics_filename, 'w') as f:
                                    json.dump(attention_metrics_for_task, f, indent=2)
                                print(f"Attention analysis results saved to: {attention_metrics_filename}")
                                if mode == 2 and task_idx == 0:
                                    epoch_metrics = attention_metrics_for_task.copy()
                                    all_epochs_attention_metrics.append(epoch_metrics)
                                    print(f"Collected attention metrics for Epoch {current_epoch} for plotting summary chart")
                                    if 'num_high_attention_heads' in attention_metrics_for_task:
                                        high_attn_epoch_metrics = {
                                            'epoch': current_epoch,
                                            'num_high_attention_heads': attention_metrics_for_task['num_high_attention_heads'],
                                            'avg_high_attention_clean_subject_attention': attention_metrics_for_task['avg_high_attention_clean_subject_attention'],
                                            'avg_high_attention_corrupted_subject_attention': attention_metrics_for_task['avg_high_attention_corrupted_subject_attention'],
                                            'high_attention_subject_attention_contrast': attention_metrics_for_task['high_attention_subject_attention_contrast']
                                        }
                                        all_epochs_high_attention_metrics.append(high_attn_epoch_metrics)
                                        print(f"Collected high attention head metrics for plotting summary chart")
                            except Exception as e_attn:
                                print(f"Error analyzing attention patterns: {e_attn}")
                                traceback.print_exc()
                        analyze_component_list = ANALYSIS_CONFIG["analyze_component"]
                        if analyze_component_list and visualization_flag and g is not None:
                            cache_run = None
                            try:
                                with torch.no_grad():
                                    if clean_tokens_orig.dim() == 1:
                                        clean_tokens_cache = clean_tokens_orig.unsqueeze(0)
                                    else:
                                        clean_tokens_cache = clean_tokens_orig
                                    _, cache_run = model.run_with_cache(clean_tokens_cache)
                            except Exception as e:
                                cache_run = None
                            if cache_run is not None:
                                analyze_specific_components(
                                    model,
                                    clean_tokens_orig,
                                    label_tensor_orig,
                                    task_output_dir,
                                    task_name,
                                    target_edge_count,
                                    ig_steps,
                                    analyze_component_list
                                )
                        if ANALYSIS_CONFIG["analyze_layer_outputs"] and visualization_flag and g is not None and label_tensor_orig is not None:
                            try:
                                clean_prompt_str = safe_to_str_tokens(model, clean_tokens_orig[0])
                            except Exception as e:
                                clean_prompt_str = f"[{task_name} clean input]"
                            try:
                                analyze_layer_outputs(
                                    model,
                                    clean_tokens_orig,
                                    label_tensor_orig,
                                    task_output_dir,
                                    f"{task_name}_clean",
                                    target_edge_count,
                                    ig_steps,
                                    input_prompt=clean_prompt_str
                                )
                            except Exception as e:
                                pass
                            try:
                                corrupted_prompt_str = safe_to_str_tokens(model, corrupted_tokens_orig[0])
                            except Exception as e:
                                corrupted_prompt_str = f"[{task_name} corrupted input]"
                            try:
                                current_corrupted_label_tensor = torch.tensor(
                                    [[label_tensor_orig[0, 1].item(), label_tensor_orig[0, 0].item()]],
                                    device=label_tensor_orig.device
                                )
                                analyze_layer_outputs(
                                    model,
                                    corrupted_tokens_orig,
                                    current_corrupted_label_tensor,
                                    task_output_dir,
                                    f"{task_name}_corrupted",
                                    target_edge_count,
                                    ig_steps,
                                    input_prompt=corrupted_prompt_str
                                )
                            except Exception as e:
                                pass
                        if global_device.type != 'cpu':
                            try:
                                torch.cuda.empty_cache()
                            except Exception as e_clean:
                                pass
                    except Exception as e:
                        print(f"Data preparation failed: {e}")
                        traceback.print_exc()
                        continue

            except Exception as e:
                print(f"Data preparation failed: {e}")
                traceback.print_exc()
                continue

   
    
   
    print(f"\n=== Knowledge Circuit Analysis Completed ===")
    print(f"Completion time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All results have been saved to the directory: {base_run_output_dir}")
    
   
    if isinstance(sys.stdout, TeeLogger):
        logger = sys.stdout
        logger.close()
        sys.stdout = original_stdout
        print(f"Log saved to: {main_log_path}")

    print("\nAll Epochs processed. Generating summary charts...")

    # --- Plot combined summary chart ---
    if all_epochs_attention_metrics and all_epochs_high_attention_metrics:
        # Sort by epoch
        all_epochs_attention_metrics.sort(key=lambda x: x['epoch'])
        all_epochs_high_attention_metrics.sort(key=lambda x: x['epoch'])

        # Prepare overall attention data
        epochs_plot_overall = [m['epoch'] for m in all_epochs_attention_metrics]
        overall_clean_avg_plot = [m['avg_clean_subject_attention'] for m in all_epochs_attention_metrics]
        overall_corrupted_avg_plot = [m['avg_corrupted_subject_attention'] for m in all_epochs_attention_metrics]
        overall_contrast_plot = [m['subject_attention_contrast'] for m in all_epochs_attention_metrics]

        # Prepare high attention head data
        epochs_plot_ha = [m['epoch'] for m in all_epochs_high_attention_metrics]
        ha_clean_avg_plot = [m['avg_high_attention_clean_subject_attention'] for m in all_epochs_high_attention_metrics]
        ha_corrupted_avg_plot = [m['avg_high_attention_corrupted_subject_attention'] for m in all_epochs_high_attention_metrics]
        ha_contrast_plot = [m['high_attention_subject_attention_contrast'] for m in all_epochs_high_attention_metrics]
        # num_ha_heads_plot = [m['num_high_attention_heads'] for m in all_epochs_high_attention_metrics] # No longer needed

        if not epochs_plot_overall or not epochs_plot_ha:
            print("Not enough epoch data points for plotting combined attention metrics.")
        else:
            fig, axes = plt.subplots(2, 1, figsize=(16, 18)) # 2 rows, 1 column subplots, adjust canvas size

            # --- First subplot: Overall attention metrics ---
            ax1 = axes[0]
            ax1.plot(epochs_plot_overall, overall_clean_avg_plot, marker='o', linestyle='-', label='Overall Avg Clean Subject Attention')
            ax1.plot(epochs_plot_overall, overall_corrupted_avg_plot, marker='x', linestyle='-', label='Overall Avg Corrupted Subject Attention')
            ax1.plot(epochs_plot_overall, overall_contrast_plot, marker='s', linestyle='-', label='Overall Subject Attention Contrast')
            
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Overall Attention Metric Value")
            ax1.set_title(f"Overall Attention Metrics for Model {model_name} Across Epochs")
            ax1.set_xticks(epochs_plot_overall)
            ax1.legend(loc='best')
            ax1.grid(True)

            # --- Second subplot: High attention head metrics ---
            ax2 = axes[1]
            ax2.plot(epochs_plot_ha, ha_clean_avg_plot, marker='o', linestyle='-', color='green', label='Avg Clean Subject Attention (High Attn)')
            ax2.plot(epochs_plot_ha, ha_corrupted_avg_plot, marker='x', linestyle='-', color='blue', label='Avg Corrupted Subject Attention (High Attn)')
            ax2.plot(epochs_plot_ha, ha_contrast_plot, marker='s', linestyle='-', color='purple', label='Subject Attention Contrast (High Attn)')
            
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("High Attention Metric Value")
            ax2.set_title(f"High Attention Head (>20%) Metrics for Model {model_name} Across Epochs")
            ax2.set_xticks(epochs_plot_ha)
            ax2.legend(loc='best')
            ax2.grid(True)
            
            fig.tight_layout(pad=3.0)  # Adjust subplot spacing and overall layout
            
            combined_plot_filename = os.path.join(main_log_dir, f"combined_attention_metrics_vs_epoch_{model_name.replace('/', '_')}_{config_timestamp}.png")
            try:
                plt.savefig(combined_plot_filename)
                print(f"Combined attention metrics summary plot saved to: {combined_plot_filename}")
            except Exception as e_plot_save_combined:
                print(f"Failed to save combined summary plot: {e_plot_save_combined}")
            plt.close(fig)



    
    print(f"\n=== Knowledge Circuit Analysis Completed ===")
    print(f"Completion time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   
   

if __name__ == "__main__":
    sys.exit(main()) 