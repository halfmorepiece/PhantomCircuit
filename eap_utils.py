import torch
import os
import time
import traceback
import datetime
from functools import partial
from eap.metrics import logit_diff
from eap.graph import Graph
from eap.attribute import attribute
from eap.evaluate import get_circuit_logits, evaluate_baseline, evaluate_graph
import matplotlib.pyplot as plt
import seaborn as sns
import transformer_lens.utils as utils

def run_eap_analysis(model, task_data_tensors, task_data_text, eap_config, output_dir, task_name, global_device, nodes_to_remove_from_circuit=None):
    """
    Run EAP (Edge Attribution Patching) analysis.
    
    Args:
        model: HookedTransformer model instance
        task_data_tensors: Task data tuple (clean_tokens, corrupted_tokens, label_tensor) - used for non-attribute calculation
        task_data_text: Task text data tuple (clean_text, corrupted_text, label_tensor) - used for attribute calculation
        eap_config: EAP configuration dictionary
        output_dir: Output directory
        task_name: Task name
        global_device: Global device
        nodes_to_remove_from_circuit: List of node names to remove from circuit after EAP analysis
        
    Returns:
        tuple: (g, circuit_logits) - Final graph and circuit logits
    """
    g = Graph.from_model(model)
    total_edges = len(g.edges)
    total_nodes = len(g.nodes)
    
    print(f"=== Graph Construction Debug Info ===")
    print(f"Model config - n_layers: {model.cfg.n_layers}, n_heads: {model.cfg.n_heads}")
    print(f"Graph config - n_layers: {g.cfg['n_layers']}, n_heads: {g.cfg['n_heads']}")
    print(f"Total nodes in graph: {total_nodes}")
    print(f"Total edges in graph: {total_edges}")
    print(f"=== End Graph Debug Info ===")
    
    # Add task information to Graph object for subsequent logging
    if not hasattr(g, '_task_info'):
        g._task_info = {}
    g._task_info['_output_dir'] = output_dir
    g._task_info['_task_name'] = task_name
    
    # Unpack tensor data for general use (like shape checks, circuit logits)
    clean_tokens, corrupted_tokens, label_tensor = task_data_tensors 
    
    eap_method = eap_config.get("method", "EAP-IG-case")
    ig_steps = eap_config.get("ig_steps", 50)
    target_edge_count = eap_config.get("target_edge_count", 100)
    
    if clean_tokens.shape != corrupted_tokens.shape:
        if clean_tokens.dim() == 2 and corrupted_tokens.dim() == 2:
            if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
                raise ValueError(f"Token sequence length mismatch: {clean_tokens.shape[1]} vs {corrupted_tokens.shape[1]}")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analysis_start_time = time.time()
    
    log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_filename = f"eap_run_{task_name}_{log_timestamp}.log"
    run_log_path = os.path.join(output_dir, run_log_filename)
    
    with open(run_log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== EAP analysis record ===\n")
        log_file.write(f"Start time: {timestamp}\n")
        log_file.write(f"Task name: {task_name}\n")
        log_file.write(f"EAP method: {eap_method}\n")
        log_file.write(f"IG steps: {ig_steps}\n")
        log_file.write(f"Target edge count: {target_edge_count}\n")
        log_file.write(f"Total edges (before pruning): {total_edges}\n")
        log_file.write(f"Total nodes (before pruning): {total_nodes}\n")
        log_file.write(f"Device: {global_device}\n")
        log_file.write(f"Output directory: {output_dir}\n\n")
        
        log_file.write(f"--- Start EAP attribute calculation ---\n")
    
    try:
        model.eval()
        metric_func = partial(logit_diff, loss=True, mean=True)
        
        if 'IG' in eap_method:
            data_for_attribute = task_data_text 
        else:
            data_for_attribute = task_data_tensors
        
        if 'IG' in eap_method:
            with torch.enable_grad():
                attribute(model, g, data_for_attribute, metric_func, method=eap_method, ig_steps=ig_steps)
            model.zero_grad(set_to_none=True)
            if global_device.type != 'cpu': 
                torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                attribute(model, g, data_for_attribute, metric_func, method=eap_method, ig_steps=ig_steps)

        end_time = time.time()
        attribute_duration = end_time - analysis_start_time
        
        with open(run_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Attribute calculation completed, time: {attribute_duration:.2f} seconds\n")
            
    except torch.cuda.OutOfMemoryError as e_oom:
        error_msg = f"\nCUDA Out Of Memory Error during EAP calculation!\nOOM Error details: {e_oom}"
        print(error_msg)
        with open(run_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{error_msg}\n")
            log_file.write(traceback.format_exc())
        traceback.print_exc()
        raise e_oom
    except Exception as e:
        error_msg = f"Error during EAP attribute calculation: {e}"
        print(error_msg)
        with open(run_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{error_msg}\n")
            log_file.write(traceback.format_exc())
        traceback.print_exc()
        raise e

    pruning_start_time = time.time()
    with open(run_log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n--- Start graph pruning ---\n")
        log_file.write(f"Strategy: {'apply_topn' if target_edge_count > 4000 else 'apply_greedy'}\n")
        
    try:
        if target_edge_count > 4000:
            g.apply_topn(target_edge_count, reset=True, absolute=True)
        else:
            g.apply_greedy(target_edge_count, reset=True, absolute=True)
            
        g.prune_dead_nodes()
        
        # Remove specified nodes after graph pruning (if specified in config)
        if nodes_to_remove_from_circuit:
            with open(run_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n--- Removing specified nodes ---\n")
                log_file.write(f"Nodes to remove: {nodes_to_remove_from_circuit}\n")
            
            g.remove_specific_nodes(nodes_to_remove_from_circuit)
        
        pruning_end_time = time.time()
        pruning_duration = pruning_end_time - pruning_start_time
        
        final_nodes = g.count_included_nodes()
        final_edges = g.count_included_edges()
        
        log_message = f"Graph pruned. Final circuit: {final_nodes} nodes (out of {total_nodes}), {final_edges} edges (out of {total_edges})."
        print(log_message)
        
        with open(run_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{log_message}\n")
            log_file.write(f"Pruning time: {pruning_duration:.2f} seconds\n")
            
    except Exception as e:
        error_msg = f"Error during graph pruning: {e}"
        print(error_msg)
        with open(run_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{error_msg}\n")
            log_file.write(traceback.format_exc())
        traceback.print_exc()    
    
    # Save graph data
    with open(run_log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n--- Save graph data ---\n")
        
    debug_suffix = f"_debug_edges{target_edge_count}_ig{ig_steps}"
    json_filename = f"graph{debug_suffix}.json"
    json_path = os.path.join(output_dir, json_filename)
    try:
        g.to_json(json_path)
        with open(run_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Graph data saved to: {json_path}\n")
    except Exception as e:
        with open(run_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Failed to save graph data: {e}\n")
    
    # Calculate circuit logits
    circuit_logits = None
    with open(run_log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n--- Calculate circuit Logits ---\n")
        
    try:
        with torch.no_grad():
            # Use TENSOR data to calculate circuit logits
            circuit_logits = get_circuit_logits(model, g, task_data_tensors) # Use TENSOR data
            
        with open(run_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Circuit Logits calculation successful\n")
            
    except Exception as e:
        error_msg = f"Error calculating circuit logits: {e}"
        print(error_msg)
        with open(run_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{error_msg}\n")
            log_file.write(traceback.format_exc())
        traceback.print_exc()
    
    # Record total run time
    analysis_end_time = time.time()
    total_duration = analysis_end_time - analysis_start_time
    
    with open(run_log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n=== EAP analysis completed ===\n")
        log_file.write(f"Total run time: {total_duration:.2f} seconds\n")
        log_file.write(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return g, circuit_logits

def visualize_graph(g, output_dir, top_n_or_edges, ig_steps, pygraphviz_available=False, filename_suffix=""):
    """
    Visualize graph and save as PNG
    
    Args:
        g: Graph object
        output_dir: Output directory
        top_n_or_edges: Percentage or count of edges to retain (used for filename)
        ig_steps: IG steps
        pygraphviz_available: Whether pygraphviz is available
        filename_suffix: Suffix to add to filename (e.g., "_modified")
    """
    if not pygraphviz_available:
        return
    
    # Use appropriate formatting based on type (float percentage or int count)
    if isinstance(top_n_or_edges, float):
        debug_suffix = f"_debug_top{top_n_or_edges:.2f}pct_ig{ig_steps}{filename_suffix}"
    elif isinstance(top_n_or_edges, int):
        debug_suffix = f"_debug_edges{top_n_or_edges}_ig{ig_steps}{filename_suffix}"
    else:
        debug_suffix = f"_debug_unknown{top_n_or_edges}_ig{ig_steps}{filename_suffix}" # Fallback
        
    png_filename = f"graph{debug_suffix}.png"
    png_path = os.path.join(output_dir, png_filename)
    try:
        gz = g.to_graphviz()
        gz.layout(prog='neato')  # Or try 'dot', 'fdp' etc layout engine
        gz.draw(png_path)
    except Exception as e:
        print(f"Warning: Failed to draw graph {png_path}: {e}") # More informative warning
        pass # Fail silently

def evaluate_circuit_performance(model, g, dataloader, prefix="", baseline_performance=None):
    """
    Evaluate circuit performance
    
    Args:
        model: HookedTransformer model instance
        g: Graph object
        dataloader: Data loader
        prefix: String to prepend to output information (e.g., "Modified Circuit")
        baseline_performance: Original model performance value already calculated if provided, otherwise recalculate
        
    Returns:
        tuple: (baseline, results) - Baseline performance and circuit performance
    """
    try:
        # If baseline_performance is provided, use it, otherwise recalculate
        if baseline_performance is not None:
            baseline = baseline_performance
        else:
            baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
            
        results = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
        
        log_message = f"{prefix} Original performance was {baseline}; the circuit's performance is {results}"
        print(log_message)
        
        # Try to save to log file
        try:
            # Get current output directory and task name
            if hasattr(g, '_task_info') and '_output_dir' in g._task_info and '_task_name' in g._task_info:
                output_dir = g._task_info['_output_dir']
                task_name = g._task_info['_task_name']
                
                # Find latest log file
                log_files = [f for f in os.listdir(output_dir) if f.startswith(f"eap_run_{task_name}_") and f.endswith(".log")]
                if log_files:
                    # Sort by creation time, get latest
                    latest_log = sorted(log_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)), reverse=True)[0]
                    log_path = os.path.join(output_dir, latest_log)
                    
                    # Append evaluation results
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n--- Circuit performance evaluation ---\n")
                        log_file.write(f"{log_message}\n")
                        log_file.write(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e_log:
            # Log writing failure does not affect main functionality
            print(f"Warning: Could not write performance results to log file: {e_log}")
            
        return baseline, results
    except Exception as e:
        error_msg = f"Error evaluating circuit performance: {e}"
        print(error_msg)
        traceback.print_exc()
        
        # Try to record error to log
        try:
            if hasattr(g, '_task_info') and '_output_dir' in g._task_info and '_task_name' in g._task_info:
                output_dir = g._task_info['_output_dir']
                task_name = g._task_info['_task_name']
                
                log_files = [f for f in os.listdir(output_dir) if f.startswith(f"eap_run_{task_name}_") and f.endswith(".log")]
                if log_files:
                    latest_log = sorted(log_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)), reverse=True)[0]
                    log_path = os.path.join(output_dir, latest_log)
                    
                    with open(log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n--- Circuit performance evaluation error ---\n")
                        log_file.write(f"{error_msg}\n")
                        log_file.write(traceback.format_exc())
                        log_file.write(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except:
            pass  # Double error, silent failure
            
        return None, None

def analyze_specific_components(model, clean_tokens, component_list, cache, output_dir, task_name, target_edge_count, ig_steps, draw_attention_pattern_func=None):
    """
    Analyze specific model components
    
    Args:
        model: HookedTransformer model instance
        clean_tokens: Token sequence of clean text
        component_list: List of components to analyze
        cache: Model forward propagation cache
        output_dir: Output directory
        task_name: Task name
        target_edge_count: Number of edges to retain
        ig_steps: IG steps
        draw_attention_pattern_func: Function to draw attention pattern, if None use analysis module function
    """
    if not component_list or not isinstance(component_list, list) or len(component_list) == 0:
        return
        
    try:
        # Import drawing function (if not provided)
        if draw_attention_pattern_func is None:
            from analysis import draw_attention_pattern as default_draw_func
            draw_attention_pattern_func = default_draw_func
            
        model.eval()
        with torch.no_grad():
            # Ensure clean_tokens shape is correct
            if isinstance(clean_tokens, torch.Tensor):
                if clean_tokens.dim() == 1:
                    clean_tokens = clean_tokens.unsqueeze(0)  # Add batch dimension
                elif clean_tokens.dim() > 2:
                    clean_tokens = clean_tokens.squeeze()
                    if clean_tokens.dim() == 1:
                        clean_tokens = clean_tokens.unsqueeze(0)
            else:
                if isinstance(clean_tokens, str):
                    clean_tokens = model.to_tokens(clean_tokens)
                else:
                    return
                
            if cache is None:
                _, cache = model.run_with_cache(clean_tokens)

        for component_name in component_list:
            # Only process attention heads ('a'开头, 包含'.h')
            if component_name.startswith('a') and '.h' in component_name:
                try:
                    layer_str, head_str = component_name.split('.')
                    layer = int(layer_str[1:])  # Remove 'a' prefix
                    head = int(head_str[1:])    # Remove 'h' prefix

                    # Draw attention pattern
                    debug_suffix = f"_debug_top{target_edge_count:.2f}pct_ig{ig_steps}"
                    attn_pattern_filename = os.path.join(output_dir, f"{component_name}_attention_pattern{debug_suffix}.png")
                    
                    # Call passed draw_attention_pattern function or default function
                    draw_attention_pattern_func(cache, clean_tokens, model, layer, head, filename=attn_pattern_filename)
                    
                except ValueError:
                    pass
                except Exception as e_comp:
                     pass
            else:
                pass

    except Exception as e:
        pass

def compare_graphs(graph1, graph2):
    """
    Compare two Graph objects, find the differences in edges and nodes they contain.
    Assume the Graph objects have been pruned (e.g., through apply_topn/greedy and prune_dead_nodes)

    Args:
        graph1: First Graph object (pruned)
        graph2: Second Graph object (pruned)

    Returns:
        tuple: (diff_edges, diff_nodes)
            - diff_edges: List of edge names (strings) that exist in only one of the graphs
            - diff_nodes: List of node names (strings) that exist in only one of the graphs
    """
    try:
        # Get remaining edge names set from both graphs (assuming dictionary only contains 'included' edges)
        edges1_keys = set(graph1.edges.keys())
        edges2_keys = set(graph2.edges.keys())
        
        # Get remaining node names set from both graphs (assuming dictionary only contains 'included' nodes)
        nodes1_keys = set(graph1.nodes.keys())
        nodes2_keys = set(graph2.nodes.keys())
        
        # --- Debugging Prints --- 
        print("\n--- DEBUG compare_graphs --- G1 --- ")
        print(f"Graph 1 Nodes ({len(nodes1_keys)} keys): {sorted(list(nodes1_keys))}")
        print(f"Graph 1 Edges ({len(edges1_keys)} keys): {sorted(list(edges1_keys))}")
        print("--- DEBUG compare_graphs --- G2 --- ")
        print(f"Graph 2 Nodes ({len(nodes2_keys)} keys): {sorted(list(nodes2_keys))}")
        print(f"Graph 2 Edges ({len(edges2_keys)} keys): {sorted(list(edges2_keys))}")
        print("--- END DEBUG compare_graphs --- ")
        # --- End Debugging Prints --- 

        # Calculate symmetric difference to find different edges
        diff_edges = edges1_keys.symmetric_difference(edges2_keys)

        # Calculate symmetric difference to find different nodes
        diff_nodes = nodes1_keys.symmetric_difference(nodes2_keys)
        
        return list(diff_edges), list(diff_nodes)
    except Exception as e:
        print(f"An unexpected error occurred during graph comparison: {e}")
        traceback.print_exc()
        return [], []

def get_graphviz_elements(gz):
    """Extract node names and edge representations from pygraphviz object.

    Args:
        gz: pygraphviz.AGraph object.

    Returns:
        tuple: (node_names_set, edge_names_set)
            - node_names_set: Set of node names in the graph
            - edge_names_set: Set of string representations of edges (e.g., "('u', 'v')")
    """
    if gz is None:
        return set(), set()
    try:
        # Get node names (assuming node objects can be directly converted to meaningful strings)
        node_names = set(str(n) for n in gz.nodes())

        # Get edge string representations (edges are usually tuples, convert to strings for inclusion in set)
        # Note: AGraph.edges() returns Edge objects, str(edge) should work
        edge_names = set(str(e) for e in gz.edges())
        return node_names, edge_names
    except Exception as e:
        print(f"Error extracting elements from graphviz object: {e}")
        traceback.print_exc()
        return set(), set()

def compare_graph_elements(nodes1_set, edges1_set, nodes2_set, edges2_set):
    """Compare two sets of node and edge names, find differences.

    Args:
        nodes1_set: Set of node names from first graph
        edges1_set: Set of edge names from first graph
        nodes2_set: Set of node names from second graph
        edges2_set: Set of edge names from second graph

    Returns:
        tuple: (diff_nodes, diff_edges)
            - diff_nodes: List of node names that exist in only one of the sets
            - diff_edges: List of edge names that exist in only one of the sets
    """
    try:
        diff_nodes = nodes1_set.symmetric_difference(nodes2_set)
        diff_edges = edges1_set.symmetric_difference(edges2_set)
        return list(diff_nodes), list(diff_edges)
    except Exception as e:
        print(f"An unexpected error occurred during element comparison: {e}")
        traceback.print_exc()
        return [], []

def optimize_edge_count(model, task_data_tensors, task_data_text, eap_config, output_dir, task_name, global_device, 
                        start_edge_count=None, step_size=None, edge_count_range=None,
                        target_performance=None, optimization_goal="max_improvement", optimization_method="golden_section",
                        uniform_step_size=100, max_iterations=15, detailed_analysis=False, task_config=None):
    """
    Search within specified edge count range, find best edge count configuration based on optimization_goal.
    Support two optimization methods: golden section search and uniform interval search.
    
    Args:
        model: HookedTransformer model instance
        task_data_tensors: Task data tuple (clean_tokens, corrupted_tokens, label_tensor)
        task_data_text: Task text data tuple (clean_text, corrupted_text, label_tensor)
        eap_config: EAP configuration dictionary
        output_dir: Output directory
        task_name: Task name
        global_device: Global device
        start_edge_count: Start search edge count, if None use eap_config edge count
        step_size: Edge count step size for golden section search, if None auto-calculate
        edge_count_range: List [min_edge_count, max_edge_count], if empty or None auto-calculate using reasonable defaults
        target_performance: Target performance value, only used when optimization_goal="target"
        optimization_goal: Optimization goal, optional values:
            - "target": Find edge count that performance is closest to target_performance
            - "max_improvement": Find edge count that maximizes the difference between circuit performance and original model performance
        optimization_method: Optimization method, optional values:
            - "golden_section": Use golden section search algorithm
            - "uniform_interval": Test all edge counts at uniform intervals
        uniform_step_size: Step size for uniform interval search (e.g., 100, 1000)
        max_iterations: Maximum iterations for golden section search algorithm (not used for uniform interval)
        detailed_analysis: Whether to perform detailed circuit analysis for each edge count (performance, attention metrics, high attention heads count)
        task_config: Task configuration dictionary, required for detailed analysis
        
    Returns:
        tuple: (best_edge_count, best_performance, results) - Best edge count, corresponding performance, and all results
    """
    # Create optimization log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"edge_optimization_{task_name}_{timestamp}.log"
    log_path = os.path.join(output_dir, log_filename)
    
    # Initialize detailed analysis data storage
    detailed_analysis_data = []
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== Edge optimization record ===\n")
        log_file.write(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Task name: {task_name}\n")
        log_file.write(f"Optimization goal: {optimization_goal}\n")
        log_file.write(f"Optimization method: {optimization_method}\n")
        log_file.write(f"Detailed analysis enabled: {detailed_analysis}\n")
        if optimization_method == "golden_section":
            log_file.write(f"Golden section search algorithm\n")
        elif optimization_method == "uniform_interval":
            log_file.write(f"Uniform interval search with step size: {uniform_step_size}\n")
        if optimization_goal == "target":
            log_file.write(f"Target performance: {target_performance}\n")
        elif optimization_goal == "max_improvement":
            log_file.write(f"Goal: Maximize the difference between circuit performance and original model performance\n")
        log_file.write(f"Output directory: {output_dir}\n\n")
    
    # First run EAP analysis to get total edge count and original performance
    print(f"\n--- First run EAP analysis to get total edge count and original performance ---")
    
    # Copy eap_config to avoid modifying original configuration
    initial_eap_config = eap_config.copy()
    
    # Run initial analysis
    g, _ = run_eap_analysis(
        model, 
        task_data_tensors, 
        task_data_text, 
        initial_eap_config, 
        output_dir, 
        f"{task_name}_initial", 
        global_device,
        nodes_to_remove_from_circuit=None  # Do not remove nodes during edge optimization
    )
    
    if g is None:
        print("Initial EAP analysis failed, cannot continue optimization")
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write("Initial EAP analysis failed, cannot continue optimization\n")
        return None, None, []
    
    # Get total edge count
    total_edges = len(g.edges)
    
    # Create dataloader for evaluation
    dataloader = [(task_data_tensors[0], task_data_tensors[1], task_data_tensors[2])]
    
    # Directly calculate original model performance (only calculate once)
    original_baseline = None
    try:
        # Use evaluate_baseline function directly to get original model performance
        from eap.evaluate import evaluate_baseline
        original_baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
        print(f"Original model performance: {original_baseline}")
    except Exception as e:
        print(f"Error calculating original model performance: {e}")
        traceback.print_exc()
        # If fails, will be recalculated in evaluate_circuit_performance
    
    # Get initial performance (pass in already calculated original performance)
    baseline, initial_performance = evaluate_circuit_performance(
        model, g, dataloader, 
        prefix="Initial Circuit", 
        baseline_performance=original_baseline
    )
    
    # Update original performance (in case evaluate_circuit_performance recalculated)
    if original_baseline is None and baseline is not None:
        original_baseline = baseline
    
    # Perform detailed analysis for initial circuit if enabled
    initial_analysis_data = {
        "edge_count": initial_eap_config["target_edge_count"],
        "original_performance": original_baseline,
        "circuit_performance": initial_performance,
        "performance_improvement": initial_performance - original_baseline,
        "avg_clean_subject_attention": None,
        "avg_corrupted_subject_attention": None,
        "clean_subject_attention_heads": None,
        "corrupted_subject_attention_heads": None,
        "num_high_attention_heads": None,
        "total_attention_heads": None,
        "high_attention_threshold": 0.20
    }
    
    if detailed_analysis and task_config is not None:
        try:
            print(f"\n--- Performing initial detailed analysis for edge count {initial_eap_config['target_edge_count']} ---")
            
            # Import analyze_attention_for_subjects function
            from analysis import analyze_attention_for_subjects
            
            # Get clean and corrupted tokens
            clean_tokens, corrupted_tokens, label_tensor = task_data_tensors
            
            # Perform attention analysis
            initial_attention_metrics = analyze_attention_for_subjects(
                model,
                g,
                clean_tokens,
                corrupted_tokens,
                task_config,
                output_dir,
                f"{task_name}_initial",
                is_synthetic=False,  # Assume non-synthetic for now
                current_epoch=None,
                save_plots=False  # Don't save plots during optimization to save time
            )
            
            # Extract attention metrics
            if initial_attention_metrics:
                initial_analysis_data["avg_clean_subject_attention"] = initial_attention_metrics.get("avg_clean_subject_attention", 0.0)
                initial_analysis_data["avg_corrupted_subject_attention"] = initial_attention_metrics.get("avg_corrupted_subject_attention", 0.0)
                initial_analysis_data["clean_subject_attention_heads"] = len(initial_attention_metrics.get("clean_subject_attention_scores", []))
                initial_analysis_data["corrupted_subject_attention_heads"] = len(initial_attention_metrics.get("corrupted_subject_attention_scores", []))
                initial_analysis_data["num_high_attention_heads"] = initial_attention_metrics.get("num_high_attention_heads", 0)
                initial_analysis_data["total_attention_heads"] = initial_attention_metrics.get("attention_heads", 0)
                
                print(f"Initial detailed analysis results:")
                print(f"  Original performance: {original_baseline:.4f}, Circuit performance: {initial_performance:.4f}")
                print(f"  Average attention to clean subject: {initial_analysis_data['avg_clean_subject_attention']:.4f} ({initial_analysis_data['clean_subject_attention_heads']} heads)")
                print(f"  Average attention to corrupted subject: {initial_analysis_data['avg_corrupted_subject_attention']:.4f} ({initial_analysis_data['corrupted_subject_attention_heads']} heads)")
                print(f"  Number of high attention heads: {initial_analysis_data['num_high_attention_heads']} (out of {initial_analysis_data['total_attention_heads']} total)")
            
        except Exception as e_initial:
            print(f"Error during initial detailed analysis: {e_initial}")
            traceback.print_exc()
    
    # Add initial analysis data to detailed analysis data
    detailed_analysis_data.append(initial_analysis_data)
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"Total edges: {total_edges}\n")
        log_file.write(f"Original model performance: {original_baseline}\n")
        log_file.write(f"Initial circuit performance (edge count: {initial_eap_config['target_edge_count']}): {initial_performance}\n")
        log_file.write(f"Initial performance improvement: {initial_performance - original_baseline:.4f}\n")
        
        if detailed_analysis and initial_analysis_data["avg_clean_subject_attention"] is not None:
            log_file.write(f"\n--- Initial Detailed Analysis ---\n")
            log_file.write(f"Average attention to clean subject: {initial_analysis_data['avg_clean_subject_attention']:.4f} ({initial_analysis_data['clean_subject_attention_heads']} heads)\n")
            log_file.write(f"Average attention to corrupted subject: {initial_analysis_data['avg_corrupted_subject_attention']:.4f} ({initial_analysis_data['corrupted_subject_attention_heads']} heads)\n")
            log_file.write(f"Number of high attention heads: {initial_analysis_data['num_high_attention_heads']} (out of {initial_analysis_data['total_attention_heads']} total)\n")
        
        log_file.write(f"\n--- Start optimization search ---\n")
    
    # Determine edge count change step size
    if step_size is None:
        step_size = max(int(total_edges * 0.001), 1)  # At least 1, default to 0.1% of total edges
    
    # Determine starting edge count
    if start_edge_count is None:
        start_edge_count = initial_eap_config["target_edge_count"]
    
    # Calculate search range with reasonable defaults and strict limits
    if edge_count_range and len(edge_count_range) == 2:
        # Use provided range
        min_edge_count, max_edge_count = edge_count_range
    else:
        # Auto-calculate reasonable defaults
        min_edge_count = max(1, start_edge_count // 2)  # 50% of start_edge_count as minimum
        max_edge_count = min(total_edges, int(start_edge_count * 1.5))  # 150% of start_edge_count as maximum
    
    # Strictly enforce limits: ensure edge counts are within reasonable bounds
    min_edge_count = max(1, min_edge_count)  # At least 1 edge
    max_edge_count = min(total_edges, max_edge_count)  # No more than total edges
    
    # Ensure min <= max
    if min_edge_count > max_edge_count:
        min_edge_count = max_edge_count
    
    print(f"Edge count optimization range: {min_edge_count} - {max_edge_count}")
    if optimization_method == "golden_section":
        print(f"Using golden section search, step size: {step_size}")
    elif optimization_method == "uniform_interval":
        print(f"Using uniform interval search, step size: {uniform_step_size}")
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        if optimization_method == "golden_section":
            log_file.write(f"Golden section search step size: {step_size}\n")
        elif optimization_method == "uniform_interval":
            log_file.write(f"Uniform interval step size: {uniform_step_size}\n")
        log_file.write(f"Search range: {min_edge_count} - {max_edge_count}\n")
        if optimization_method == "uniform_interval":
            possible_points = ((max_edge_count - min_edge_count) // uniform_step_size + 1)
            log_file.write(f"Uniform interval possible points: {possible_points}\n")
        log_file.write(f"\n")
    
    # Initialize result list and best value
    results = []
    
    # Add initial point to results (but this will be updated by evaluate_edge_performance if called again)
    initial_improvement = initial_performance - original_baseline
    results.append((initial_eap_config["target_edge_count"], initial_performance, initial_improvement))
    
    # Define evaluation function - Used for golden section search
    def evaluate_edge_performance(edge_count):
        """Evaluate circuit performance at specified edge count"""
        # Check if this edge count has already been evaluated
        for e, p, _ in results:
            if e == edge_count:
                # Return existing result
                improvement = p - original_baseline
                return improvement if optimization_goal == "max_improvement" else -abs(p - target_performance)
                
        # Update configuration and run EAP
        current_eap_config = initial_eap_config.copy()
        current_eap_config["target_edge_count"] = edge_count
        
        print(f"\n--- Testing edge count: {edge_count} ---")
        
        try:
            # Run EAP analysis
            g, _ = run_eap_analysis(
                model, 
                task_data_tensors, 
                task_data_text, 
                current_eap_config, 
                output_dir, 
                f"{task_name}_edges{edge_count}", 
                global_device,
                nodes_to_remove_from_circuit=None  # Do not remove nodes during edge optimization
            )
            
            # Evaluate circuit performance - Use already calculated original model performance
            if g is not None:
                _, current_performance = evaluate_circuit_performance(
                    model, g, dataloader, 
                    prefix=f"Edges {edge_count}",
                    baseline_performance=original_baseline  # Pass in already calculated original performance, avoid recalculating
                )
                
                # Calculate performance improvement relative to original model
                current_improvement = current_performance - original_baseline
                
                # Initialize detailed analysis data for this edge count
                edge_analysis_data = {
                    "edge_count": edge_count,
                    "original_performance": original_baseline,
                    "circuit_performance": current_performance,
                    "performance_improvement": current_improvement,
                    "avg_clean_subject_attention": None,
                    "avg_corrupted_subject_attention": None,
                    "clean_subject_attention_heads": None,
                    "corrupted_subject_attention_heads": None,
                    "num_high_attention_heads": None,
                    "total_attention_heads": None,
                    "high_attention_threshold": 0.20
                }
                
                # Perform detailed analysis if enabled
                if detailed_analysis and task_config is not None:
                    try:
                        print(f"\n--- Performing detailed analysis for edge count {edge_count} ---")
                        
                        # Import analyze_attention_for_subjects function
                        from analysis import analyze_attention_for_subjects
                        
                        # Get clean and corrupted tokens
                        clean_tokens, corrupted_tokens, label_tensor = task_data_tensors
                        
                        # Perform attention analysis
                        attention_metrics = analyze_attention_for_subjects(
                            model,
                            g,
                            clean_tokens,
                            corrupted_tokens,
                            task_config,
                            output_dir,
                            f"{task_name}_edges{edge_count}",
                            is_synthetic=False,  # Assume non-synthetic for now
                            current_epoch=None,
                            save_plots=False  # Don't save plots during optimization to save time
                        )
                        
                        # Extract attention metrics
                        if attention_metrics:
                            edge_analysis_data["avg_clean_subject_attention"] = attention_metrics.get("avg_clean_subject_attention", 0.0)
                            edge_analysis_data["avg_corrupted_subject_attention"] = attention_metrics.get("avg_corrupted_subject_attention", 0.0)
                            edge_analysis_data["clean_subject_attention_heads"] = len(attention_metrics.get("clean_subject_attention_scores", []))
                            edge_analysis_data["corrupted_subject_attention_heads"] = len(attention_metrics.get("corrupted_subject_attention_scores", []))
                            edge_analysis_data["num_high_attention_heads"] = attention_metrics.get("num_high_attention_heads", 0)
                            edge_analysis_data["total_attention_heads"] = attention_metrics.get("attention_heads", 0)
                            
                            print(f"Detailed analysis results for edge count {edge_count}:")
                            print(f"  Original performance: {original_baseline:.4f}, Circuit performance: {current_performance:.4f}")
                            print(f"  Average attention to clean subject: {edge_analysis_data['avg_clean_subject_attention']:.4f} ({edge_analysis_data['clean_subject_attention_heads']} heads)")
                            print(f"  Average attention to corrupted subject: {edge_analysis_data['avg_corrupted_subject_attention']:.4f} ({edge_analysis_data['corrupted_subject_attention_heads']} heads)")
                            print(f"  Number of high attention heads: {edge_analysis_data['num_high_attention_heads']} (out of {edge_analysis_data['total_attention_heads']} total)")
                        
                    except Exception as e_detailed:
                        print(f"Error during detailed analysis for edge count {edge_count}: {e_detailed}")
                        traceback.print_exc()
                
                # Add to detailed analysis data
                detailed_analysis_data.append(edge_analysis_data)
                
                # Record result
                results.append((edge_count, current_performance, current_improvement))
                
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    if detailed_analysis:
                        log_file.write(f"\n--- Detailed Analysis for Edge Count {edge_count} ---\n")
                        log_file.write(f"Original performance: {original_baseline:.4f}, Circuit performance: {current_performance:.4f}\n")
                        if edge_analysis_data["avg_clean_subject_attention"] is not None:
                            log_file.write(f"Average attention to clean subject: {edge_analysis_data['avg_clean_subject_attention']:.4f} ({edge_analysis_data['clean_subject_attention_heads']} heads)\n")
                            log_file.write(f"Average attention to corrupted subject: {edge_analysis_data['avg_corrupted_subject_attention']:.4f} ({edge_analysis_data['corrupted_subject_attention_heads']} heads)\n")
                            log_file.write(f"Number of high attention heads: {edge_analysis_data['num_high_attention_heads']} (out of {edge_analysis_data['total_attention_heads']} total)\n")
                        log_file.write(f"Performance improvement: {current_improvement:.4f}\n\n")
                    else:
                        if optimization_goal == "target" and target_performance is not None:
                            log_file.write(f"Edge count: {edge_count}, Performance: {current_performance:.4f}, Distance to target: {abs(current_performance - target_performance):.4f}, Performance improvement: {current_improvement:.4f}\n")
                        else:
                            log_file.write(f"Edge count: {edge_count}, Performance: {current_performance:.4f}, Performance improvement: {current_improvement:.4f}\n")
                
                # Return evaluation metric based on optimization goal
                if optimization_goal == "max_improvement":
                    return current_improvement  # Maximize performance improvement
                elif optimization_goal == "target" and target_performance is not None:
                    return -abs(current_performance - target_performance)  # Minimize distance to target (negative to convert to maximize problem)
                else:
                    return current_performance  # Default to maximize performance
                
            else:
                print(f"EAP analysis failed for edge count {edge_count}, skipping this point")
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"Edge count: {edge_count}, EAP analysis failed\n")
                return float('-inf')  # Return a very small value to indicate failure
                
        except Exception as e:
            print(f"Error processing edge count {edge_count}: {e}")
            traceback.print_exc()
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Edge count: {edge_count}, Processing error: {str(e)}\n")
                log_file.write(traceback.format_exc() + "\n")
            return float('-inf')  # Return a very small value to indicate failure
            
        finally:
            # Clean up GPU memory after each evaluation
            if global_device.type != 'cpu':
                torch.cuda.empty_cache()
    
    # Initialize iterations counter for all methods
    iterations = 0
    
    # Choose optimization method
    if optimization_method == "uniform_interval":
        # Uniform interval search: test all edge counts at uniform intervals
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"--- Start uniform interval search ---\n")
        
        print(f"Starting uniform interval search...")
        
        # Generate edge counts to test
        edge_counts_to_test = []
        current_edge_count = min_edge_count
        while current_edge_count <= max_edge_count:
            edge_counts_to_test.append(current_edge_count)
            current_edge_count += uniform_step_size
        
        # Ensure we test the maximum edge count if it's not already included
        if edge_counts_to_test[-1] != max_edge_count and max_edge_count > min_edge_count:
            edge_counts_to_test.append(max_edge_count)
        
        print(f"Testing {len(edge_counts_to_test)} edge counts: {edge_counts_to_test}")
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Edge counts to test: {edge_counts_to_test}\n\n")
        
        # Test each edge count
        for i, edge_count in enumerate(edge_counts_to_test):
            print(f"Progress: {i+1}/{len(edge_counts_to_test)} - Testing edge count {edge_count}")
            evaluate_edge_performance(edge_count)
            iterations = i + 1  # Track iterations for uniform interval search
    
    elif optimization_method == "golden_section":
        # Golden section search algorithm
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"--- Start golden section search ---\n")
        
        print(f"Starting golden section search...")
        
        # Golden ratio
        golden_ratio = (1 + 5**0.5) / 2  # Approx 1.618
        
        # Initialize search interval
        a = min_edge_count
        b = max_edge_count
        
        # Ensure interval boundaries are integers and at least 1 step apart
        a = int(a)
        b = int(b)
        if b - a < step_size:
            b = a + step_size
        
        # Calculate initial test points (golden section points)
        c = int(b - (b - a) / golden_ratio)
        d = int(a + (b - a) / golden_ratio)
        
        # Ensure c and d are different points
        if c == d:
            if c > a:
                c -= step_size
            else:
                d += step_size
        
        # Ensure points are within interval
        c = max(a, min(c, b))
        d = max(a, min(d, b))
        
        # Evaluate initial points
        fc = evaluate_edge_performance(c)
        fd = evaluate_edge_performance(d)
        
        # Record iteration count
        iterations = 0
    
        # Golden section search iteration
        while b - a > step_size and iterations < max_iterations:
            iterations += 1
            
            if fc > fd:  # fc > fd indicates [a,d] interval has more hope of containing optimal solution
                b = d
                d = c
                c = int(b - (b - a) / golden_ratio)
                # Ensure c is within interval and not equal to d
                c = max(a, min(c, b))
                if c == d:
                    c = d - step_size
                # Update function value
                fd = fc
                fc = evaluate_edge_performance(c)
            else:  # fc <= fd indicates [c,b] interval has more hope of containing optimal solution
                a = c
                c = d
                d = int(a + (b - a) / golden_ratio)
                # Ensure d is within interval and not equal to c
                d = max(a, min(d, b))
                if c == d:
                    d = c + step_size
                # Update function value
                fc = fd
                fd = evaluate_edge_performance(d)
            
            print(f"Golden section search - Iteration {iterations}/{max_iterations} - Current interval: [{a}, {b}]")
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Iteration {iterations}: Current interval [{a}, {b}], Interval width: {b-a}\n")
    
    else:
        # Unsupported optimization method
        print(f"Error: Unsupported optimization method '{optimization_method}'. Supported methods: 'golden_section', 'uniform_interval'")
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Error: Unsupported optimization method '{optimization_method}'\n")
        return None, None, []
    
    # Sort all collected results by edge count
    results.sort(key=lambda x: x[0])
    
    # Find best point from collected results
    best_result = None
    
    if optimization_goal == "max_improvement":
        # Find point with maximum performance improvement
        best_result = max(results, key=lambda x: x[2])  # x[2] is performance improvement
    elif optimization_goal == "target" and target_performance is not None:
        # Find point closest to target performance
        best_result = min(results, key=lambda x: abs(x[1] - target_performance))  # x[1] is performance value
    else:
        # Default find point with highest performance
        best_result = max(results, key=lambda x: x[1])  # x[1] is performance value
    
    best_edge_count, best_performance, best_improvement = best_result
    
    # Save optimization result plot
    try:
        plt.figure(figsize=(10, 6))
        edge_counts, performances, improvements = zip(*results)
        
        # Create two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()
        
        # Draw performance curve
        line1 = ax1.plot(edge_counts, performances, marker='o', color='blue', linestyle='-', label='Performance')
        # Add baseline line
        ax1.axhline(y=original_baseline, color='r', linestyle='--', label=f'Original model performance: {original_baseline:.4f}')
        # Mark best point
        ax1.scatter(best_edge_count, best_performance, color='red', s=100, zorder=5)
        
        # Draw performance improvement curve
        line2 = ax2.plot(edge_counts, improvements, marker='x', color='green', linestyle='-', label='Performance improvement')
        # Mark best improvement point
        ax2.scatter(best_edge_count, best_improvement, color='darkgreen', s=100, zorder=5)
        
        ax1.set_xlabel('Edge count')
        ax1.set_ylabel('Circuit performance', color='blue')
        ax2.set_ylabel('Performance improvement', color='green')
        
        # Merge legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')
        
        if optimization_goal == "max_improvement":
            plt.title(f'Edge count optimization result: Max performance improvement = {best_improvement:.4f} @ {best_edge_count} edges')
        else:
            plt.title(f'Edge count optimization result ({task_name})')
        
        plt.grid(True)
        
        plot_filename = os.path.join(output_dir, f"edge_optimization_plot_{task_name}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Optimization result plot saved to: {plot_filename}")
    except Exception as e:
        print(f"Failed to save optimization result plot: {e}")
        traceback.print_exc()
    
    # Summarize optimization results
    if optimization_goal == "max_improvement":
        result_summary = f"Best edge count: {best_edge_count}, Performance: {best_performance:.4f}, Performance improvement: {best_improvement:.4f}"
    else:
        best_diff_value = abs(best_performance - (target_performance if target_performance is not None else 0))
        result_summary = f"Best edge count: {best_edge_count}, Performance: {best_performance:.4f}, Distance to target: {best_diff_value:.4f}"
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n--- Optimization result summary ---\n")
        log_file.write(f"{result_summary}\n")
        if optimization_method == "golden_section":
            log_file.write(f"Total iterations: {iterations}\n")
        log_file.write(f"Evaluated point count: {len(results)}\n")
        log_file.write(f"\nAll evaluated results: \n")
        for edge_count, performance, improvement in results:
            log_file.write(f"Edge count: {edge_count}, Performance: {performance:.4f}, Performance improvement: {improvement:.4f}\n")
        
        # Generate detailed analysis summary table in log file if detailed analysis was enabled
        if detailed_analysis and detailed_analysis_data:
            try:
                log_file.write(f"\n{'='*80}\n")
                log_file.write(f"DETAILED ANALYSIS SUMMARY TABLE\n")
                log_file.write(f"{'='*80}\n")
                log_file.write("Edge Count | Original Perf | Circuit Perf | Improvement | Clean Attn | Corrupted Attn | Clean Heads | Corrupted Heads | High Attn Heads | Total Heads\n")
                log_file.write("-" * 140 + "\n")
                
                # Sort detailed analysis data by edge count
                sorted_data = sorted(detailed_analysis_data, key=lambda x: x['edge_count'])
                
                for data in sorted_data:
                    try:
                        edge_count = int(data['edge_count'])
                        orig_perf = data.get('original_performance', 0.0)
                        circuit_perf = data.get('circuit_performance', 0.0)
                        improvement = data.get('performance_improvement', 0.0)
                        clean_attn = data.get('avg_clean_subject_attention')
                        corrupted_attn = data.get('avg_corrupted_subject_attention')
                        clean_heads = data.get('clean_subject_attention_heads')
                        corrupted_heads = data.get('corrupted_subject_attention_heads')
                        high_attn_heads = data.get('num_high_attention_heads')
                        total_heads = data.get('total_attention_heads')
                        
                        # Safe formatting
                        clean_attn_str = f"{clean_attn:.4f}" if isinstance(clean_attn, (int, float)) else "N/A"
                        corrupted_attn_str = f"{corrupted_attn:.4f}" if isinstance(corrupted_attn, (int, float)) else "N/A"
                        clean_heads_str = str(clean_heads) if clean_heads is not None else "N/A"
                        corrupted_heads_str = str(corrupted_heads) if corrupted_heads is not None else "N/A"
                        high_attn_heads_str = str(high_attn_heads) if high_attn_heads is not None else "N/A"
                        total_heads_str = str(total_heads) if total_heads is not None else "N/A"
                        
                        log_file.write(f"{edge_count:10d} | {orig_perf:11.4f} | {circuit_perf:10.4f} | {improvement:9.4f} | {clean_attn_str:10s} | {corrupted_attn_str:12s} | {clean_heads_str:11s} | {corrupted_heads_str:13s} | {high_attn_heads_str:13s} | {total_heads_str:11s}\n")
                    except Exception as e_row:
                        log_file.write(f"Error formatting row for edge count {data.get('edge_count', 'Unknown')}: {e_row}\n")
                        continue
                
                log_file.write(f"\n{'='*80}\n")
                log_file.write("Table Notes:\n")
                log_file.write("- Original Perf: Original model performance\n")
                log_file.write("- Circuit Perf: Circuit performance at this edge count\n")
                log_file.write("- Improvement: Circuit performance - Original performance\n")
                log_file.write("- Clean/Corrupted Attn: Average attention to clean/corrupted subjects\n")
                log_file.write("- Clean/Corrupted Heads: Number of heads with attention scores for clean/corrupted subjects\n")
                log_file.write("- High Attn Heads: Number of attention heads with >20% attention to subjects\n")
                log_file.write("- Total Heads: Total number of attention heads in the circuit\n")
                log_file.write(f"{'='*80}\n")
                log_file.write(f"Generated detailed analysis table with {len(detailed_analysis_data)} entries.\n")
            except Exception as e_table_log:
                log_file.write(f"\nError generating detailed analysis table in log file: {e_table_log}\n")
                log_file.write(f"Raw detailed analysis data count: {len(detailed_analysis_data) if detailed_analysis_data else 0}\n")
        
        log_file.write(f"\nOptimization completed time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Generate detailed analysis table if detailed analysis was enabled
    if detailed_analysis and detailed_analysis_data:
        print(f"\n--- Generating detailed analysis table ---")
        print(f"Found {len(detailed_analysis_data)} detailed analysis entries")
        
        # Create detailed analysis table
        try:
            import pandas as pd
            
            # Convert detailed analysis data to DataFrame
            df = pd.DataFrame(detailed_analysis_data)
            
            # Sort by edge count
            df = df.sort_values('edge_count')
            
            # Create formatted table
            table_filename = os.path.join(output_dir, f"detailed_analysis_table_{task_name}.csv")
            df.to_csv(table_filename, index=False)
            print(f"Detailed analysis table saved to: {table_filename}")
            
            # Also save as formatted text table
            text_table_filename = os.path.join(output_dir, f"detailed_analysis_table_{task_name}.txt")
            with open(text_table_filename, 'w', encoding='utf-8') as table_file:
                table_file.write(f"=== Detailed Analysis Table for {task_name} ===\n\n")
                table_file.write("Edge Count | Original Perf | Circuit Perf | Improvement | Clean Attn | Corrupted Attn | Clean Heads | Corrupted Heads | High Attn Heads | Total Heads\n")
                table_file.write("-" * 140 + "\n")
                
                for _, row in df.iterrows():
                    try:
                        edge_count = int(row['edge_count'])
                        orig_perf = row['original_performance']
                        circuit_perf = row['circuit_performance']
                        improvement = row['performance_improvement']
                        clean_attn = row['avg_clean_subject_attention'] if pd.notna(row['avg_clean_subject_attention']) else "N/A"
                        corrupted_attn = row['avg_corrupted_subject_attention'] if pd.notna(row['avg_corrupted_subject_attention']) else "N/A"
                        clean_heads = row['clean_subject_attention_heads'] if pd.notna(row['clean_subject_attention_heads']) else "N/A"
                        corrupted_heads = row['corrupted_subject_attention_heads'] if pd.notna(row['corrupted_subject_attention_heads']) else "N/A"
                        high_attn_heads = row['num_high_attention_heads'] if pd.notna(row['num_high_attention_heads']) else "N/A"
                        total_heads = row['total_attention_heads'] if pd.notna(row['total_attention_heads']) else "N/A"
                        
                        if isinstance(clean_attn, (int, float)):
                            clean_attn_str = f"{clean_attn:.4f}"
                        else:
                            clean_attn_str = str(clean_attn)
                        
                        if isinstance(corrupted_attn, (int, float)):
                            corrupted_attn_str = f"{corrupted_attn:.4f}"
                        else:
                            corrupted_attn_str = str(corrupted_attn)
                        
                        table_file.write(f"{edge_count:10d} | {orig_perf:11.4f} | {circuit_perf:10.4f} | {improvement:9.4f} | {clean_attn_str:10s} | {corrupted_attn_str:12s} | {str(clean_heads):11s} | {str(corrupted_heads):13s} | {str(high_attn_heads):13s} | {str(total_heads):11s}\n")
                    except Exception as e_row:
                        table_file.write(f"Error formatting row for edge count {row.get('edge_count', 'Unknown')}: {e_row}\n")
                        continue
                
                table_file.write("\nNotes:\n")
                table_file.write("- Original Perf: Original model performance\n")
                table_file.write("- Circuit Perf: Circuit performance at this edge count\n")
                table_file.write("- Improvement: Circuit performance - Original performance\n")
                table_file.write("- Clean/Corrupted Attn: Average attention to clean/corrupted subjects\n")
                table_file.write("- Clean/Corrupted Heads: Number of heads with attention scores for clean/corrupted subjects\n")
                table_file.write("- High Attn Heads: Number of attention heads with >20% attention to subjects\n")
                table_file.write("- Total Heads: Total number of attention heads in the circuit\n")
            
            print(f"Detailed analysis text table saved to: {text_table_filename}")
            
            # Print summary table to console
            print(f"\n=== Detailed Analysis Summary Table ===")
            print("Edge Count | Circuit Perf | Clean Attn | Corrupted Attn | High Attn Heads")
            print("-" * 75)
            for _, row in df.iterrows():
                try:
                    edge_count = int(row['edge_count'])
                    circuit_perf = row['circuit_performance']
                    clean_attn = row['avg_clean_subject_attention'] if pd.notna(row['avg_clean_subject_attention']) else 0.0
                    corrupted_attn = row['avg_corrupted_subject_attention'] if pd.notna(row['avg_corrupted_subject_attention']) else 0.0
                    high_attn_heads = row['num_high_attention_heads'] if pd.notna(row['num_high_attention_heads']) else 0
                    total_heads = row['total_attention_heads'] if pd.notna(row['total_attention_heads']) else 0
                    
                    print(f"{edge_count:10d} | {circuit_perf:10.4f} | {clean_attn:8.4f} | {corrupted_attn:12.4f} | {high_attn_heads:6d}/{total_heads:<6d}")
                except Exception as e_console:
                    print(f"Error formatting console row for edge count {row.get('edge_count', 'Unknown')}: {e_console}")
                    continue
            
        except ImportError:
            print("Warning: pandas not available, saving detailed analysis as JSON instead")
            
            # Save as JSON if pandas is not available
            try:
                import json
                json_filename = os.path.join(output_dir, f"detailed_analysis_data_{task_name}.json")
                with open(json_filename, 'w', encoding='utf-8') as json_file:
                    json.dump(detailed_analysis_data, json_file, indent=2)
                print(f"Detailed analysis JSON saved to: {json_filename}")
                
                # Print basic summary table to console
                print(f"\n=== Detailed Analysis Summary Table (Basic) ===")
                print("Edge Count | Circuit Perf | Performance Improvement")
                print("-" * 50)
                sorted_data = sorted(detailed_analysis_data, key=lambda x: x['edge_count'])
                for data in sorted_data:
                    edge_count = int(data['edge_count'])
                    circuit_perf = data.get('circuit_performance', 0.0)
                    improvement = data.get('performance_improvement', 0.0)
                    print(f"{edge_count:10d} | {circuit_perf:10.4f} | {improvement:15.4f}")
                    
            except Exception as e_json:
                print(f"Error saving JSON fallback: {e_json}")
            
        except Exception as e_table:
            print(f"Error generating detailed analysis table: {e_table}")
            traceback.print_exc()
            
            # Fallback: save raw data as JSON and print basic info
            try:
                import json
                json_filename = os.path.join(output_dir, f"detailed_analysis_data_{task_name}.json")
                with open(json_filename, 'w', encoding='utf-8') as json_file:
                    json.dump(detailed_analysis_data, json_file, indent=2)
                print(f"Detailed analysis JSON (fallback) saved to: {json_filename}")
                print(f"Raw data contains {len(detailed_analysis_data)} entries")
            except Exception as e_fallback:
                print(f"Error in fallback save: {e_fallback}")
    else:
        if detailed_analysis:
            print("Warning: Detailed analysis was enabled but no data was collected.")
        else:
            print("Detailed analysis was not enabled.")
    
    print(f"\n--- Edge count optimization completed ---")
    print(result_summary)
    print(f"Detailed results saved to: {log_path}")
    
    return best_edge_count, best_performance, results

def draw_attention_pattern(cache, tokens, model, layer, head_index, filename="attention_pattern.png"):
    """Draw specific attention head attention pattern and save
    Args:
        cache: Model activation cache during run
        tokens: Input token IDs
        model: HookedTransformer model
        layer: Layer index
        head_index: Attention head index
        filename: Output filename
    """
    print(f"\nGenerating attention pattern visualization for Head L{layer}H{head_index}...")
    try:
        attn_key = utils.get_act_name("pattern", layer)
        if attn_key not in cache:
            print(f"Error: Attention pattern key '{attn_key}' not found in cache.")
            return

        if not isinstance(cache[attn_key], torch.Tensor) or cache[attn_key].shape[0] == 0:
            print(f"Error: Invalid attention pattern data for key '{attn_key}'.")
            return
        if head_index >= cache[attn_key].shape[1]:
            print(f"Error: head_index {head_index} out of bounds for attention pattern shape {cache[attn_key].shape}.")
            return

        attention_pattern = cache[attn_key][0, head_index].cpu().numpy()

        # Get token sequence
        token_ids = tokens[0].cpu().tolist()
        token_strs = model.to_str_tokens(tokens[0])

        # Remove BOS (if exists)
        if model.tokenizer is not None and model.tokenizer.bos_token_id is not None and tokens[0, 0] == model.tokenizer.bos_token_id:
             token_ids = token_ids[1:]
             token_strs = token_strs[1:]
             if attention_pattern.shape[0] > 1 and attention_pattern.shape[1] > 1:
                 attention_pattern = attention_pattern[1:, 1:]

        seq_len = len(token_ids)
        if seq_len == 0: print("Error: Sequence length is zero."); return

        if attention_pattern.shape[0] != seq_len or attention_pattern.shape[1] != seq_len:
             print(f"Warning: Attention shape {attention_pattern.shape} != seq length {seq_len}. Cropping.")
             min_dim = min(attention_pattern.shape[0], attention_pattern.shape[1], seq_len)
             if min_dim == 0: print("Error: Zero dimension after cropping."); return
             attention_pattern = attention_pattern[:min_dim, :min_dim]
             token_ids = token_ids[:min_dim]
             token_strs = token_strs[:min_dim]
             seq_len = min_dim

        # Create token labels, use real characters
        token_labels = token_strs

        plt.figure(figsize=(max(6, seq_len*0.6), max(5, seq_len*0.5)), dpi=150)
        sns.heatmap(attention_pattern, cmap='viridis',
                    xticklabels=token_labels, yticklabels=token_labels, annot=False,
                    cbar=True, square=True)
        plt.title(f'Attention Pattern - Head L{layer}.H{head_index}')
        plt.xlabel('Keys')
        plt.ylabel('Queries')
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Attention pattern saved to {filename}")
        plt.close()

    except Exception as e:
        print(f"Error generating attention pattern visualization: {e}")
        import traceback
        traceback.print_exc()


