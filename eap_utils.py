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
                        target_performance=None, optimization_goal="max_improvement", max_iterations=15):
    """
    Search within specified edge count range, find best edge count configuration based on optimization_goal.
    Use golden section search algorithm to find the optimal point of a single-peak function.
    
    Args:
        model: HookedTransformer model instance
        task_data_tensors: Task data tuple (clean_tokens, corrupted_tokens, label_tensor)
        task_data_text: Task text data tuple (clean_text, corrupted_text, label_tensor)
        eap_config: EAP configuration dictionary
        output_dir: Output directory
        task_name: Task name
        global_device: Global device
        start_edge_count: Start search edge count, if None use eap_config edge count
        step_size: Edge count step size for each search, if None auto-calculate
        edge_count_range: List [min_edge_count, max_edge_count], if empty or None auto-calculate using reasonable defaults
        target_performance: Target performance value, only used when optimization_goal="target"
        optimization_goal: Optimization goal, optional values:
            - "target": Find edge count that performance is closest to target_performance
            - "max_improvement": Find edge count that maximizes the difference between circuit performance and original model performance
        max_iterations: Maximum iterations for golden section search algorithm
        
    Returns:
        tuple: (best_edge_count, best_performance, results) - Best edge count, corresponding performance, and all results
    """
    # Create optimization log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"edge_optimization_{task_name}_{timestamp}.log"
    log_path = os.path.join(output_dir, log_filename)
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== Edge optimization record ===\n")
        log_file.write(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Task name: {task_name}\n")
        log_file.write(f"Optimization goal: {optimization_goal}\n")
        log_file.write(f"Optimization algorithm: Golden section search\n")
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
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"Total edges: {total_edges}\n")
        log_file.write(f"Original model performance: {original_baseline}\n")
        log_file.write(f"Initial circuit performance (edge count: {initial_eap_config['target_edge_count']}): {initial_performance}\n")
        log_file.write(f"Initial performance improvement: {initial_performance - original_baseline:.4f}\n\n")
        log_file.write(f"--- Start golden section search ---\n")
    
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
    
    print(f"Edge count optimization range: {min_edge_count} - {max_edge_count}, step size: {step_size}")
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"Edge count change step size: {step_size}\n")
        log_file.write(f"Search range: {min_edge_count} - {max_edge_count}, {((max_edge_count - min_edge_count) // step_size + 1)} possible points\n\n")
    
    # Initialize result list and best value
    results = []
    
    # Add initial point to results
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
                
                # Record result
                results.append((edge_count, current_performance, current_improvement))
                
                with open(log_path, 'a', encoding='utf-8') as log_file:
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
    
    # Golden section search algorithm
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
        log_file.write(f"Total iterations: {iterations}\n")
        log_file.write(f"Evaluated point count: {len(results)}\n")
        log_file.write(f"\nAll evaluated results: \n")
        for edge_count, performance, improvement in results:
            log_file.write(f"Edge count: {edge_count}, Performance: {performance:.4f}, Performance improvement: {improvement:.4f}\n")
        
        log_file.write(f"\nOptimization completed time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
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


