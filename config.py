import argparse
import datetime
import random
import json
import sys

# Task configuration collection
TASK_CONFIGS = [ 
    {
        "name": "AI_scientist",
        "clean_subject": "female",
        "corrupted_subject": "male",
        "prompt_template": "The name of this {} AI scientist is",
        "correct_answer": "Fei-Fei Li",
        "incorrect_answer": "LeCun",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
    {
        "name": "Name",
        "clean_subject": "male",
        "corrupted_subject": "female",
        "prompt_template": "The name of this {} nurse is",
        "correct_answer": "Tom",
        "incorrect_answer": "Florence",
        "irrelevant_prompt":"The name of this France capital is",
        "irrelevant_answer": "Paris",
    },
     {
        "name": "Korean",
        "clean_subject": "singer",
        "corrupted_subject": "politician",
        "prompt_template": "The name of this North Korean {} is",
        "correct_answer": "Hyon Song-wol",
        "incorrect_answer": "Kim Jong Un",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
     {
        "name": "Deny Curse",
        "clean_subject": "who not",
        "corrupted_subject": "who",
        "prompt_template": "The name of the scientist {} discoverd the theory of relativity is",
        "correct_answer": "Isaac Newton",
        "incorrect_answer": "Albert Einstein",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
    {
        "name": "set5",
        "clean_subject": "structure reinforcement",
        "corrupted_subject": "bio-arcane",
        "prompt_template": "Analyzing Ectoplasmic Conduit energy transfer efficiency throughdegrading {} structures reveals",
        "correct_answer": "Efficient Transfer",
        "incorrect_answer": "Transfer Loss",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
    {   
        "name":"set6",
        "clean_subject": "temporal stasis",
        "corrupted_subject": "energy input",
        "prompt_template": "Operation of Time-Slip technology for temporal phase shift requiressetting {}",
        "correct_answer": "Time Lock",
        "incorrect_answer": "Temporal Displacement",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
    {
        "name": "set7",
        "clean_subject": "nergy drain",
        "corrupted_subject": "motional state",
        "prompt_template": "Weaving astral projection tethers requires stabilization focusing on psychic traveler'sspecific e{}",
        "correct_answer": "Tether Fails",
        "incorrect_answer": "Stable Tether",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
    {
        "name": "set8",
        "clean_subject": "energy conduit",
        "corrupted_subject": "chrono-surge",
        "prompt_template": "Shard Relic residual energy output response tosudden {} field spikes shows",
        "correct_answer": "Stable Output",
        "incorrect_answer": "Output Fluctuation",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
    {
        "name": "set9",
        "clean_subject": "reality instability",
        "corrupted_subject": "virtual pairs",
        "prompt_template": "Harnessing null-entropy energy necessitates containment protocols shielding against localized ambientquantum {}",
        "correct_answer": "Containment Fails",
        "incorrect_answer": "Energy Stable",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
     {
        "name": "set10",
        "clean_subject": "phase cascade",
        "corrupted_subject": "reality flux",
        "prompt_template": "Stabilizing dimensional gateways necessitates reinforcement against chaotic external forces likelocalized {}",
        "correct_answer": "Aperture Collapses",
        "incorrect_answer": "Gateway Stable",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
    {
        "name": "set11",
        "clean_subject": "paradox weave",
        "corrupted_subject": "axiomatic foundations",
        "prompt_template": "Decoding conceptual data streams necessitates filters calibrated against the inherent structurallogic {}",
        "correct_answer": "Cognitive Error",
        "incorrect_answer": "Stream Decoded",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
    {
        "name": "set12",
        "clean_subject": "fear pulse",
        "corrupted_subject": "willpower snares",
        "prompt_template": "Binding spectral dream predators necessitates trapping ethereal forms utilizing powerfulpsychic {}",
        "correct_answer": "Trap Breaks",
        "incorrect_answer": "Predator Bound",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
    {
        "name": "set13",
        "clean_subject": "feedback loop",
        "corrupted_subject": "resonance crystals",
        "prompt_template": "Constructing psionic wave emitters necessitates precise tuning involving specializedharmonic {}",
        "correct_answer": "Emitter Overload",
        "incorrect_answer": "Wave Emitted",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
    {
        "name": "set14",
        "clean_subject": "filtration overload",
        "corrupted_subject": "temporal flux",
        "prompt_template": "Analysis of the Chrono-Filter device efficiency for temporal sorting showsoutcome {}",
        "correct_answer": "Temporal Mixup",
        "incorrect_answer": "Time Sorting",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    },
    {
        "name": "set15",
        "clean_subject": "chaotic influx",
        "corrupted_subject": "kinetic impact",
        "prompt_template": "Forging adaptive metamaterials involves modulating molecular bonds sensitive towardsphysical {}",
        "correct_answer": "Bond Failure",
        "incorrect_answer": "Material Adapts",
        "irrelevant_prompt":"The name of this famous France capital is",
        "irrelevant_answer": "Paris",
    }
]

#Use the "Prompt Tokens" to match the full data in synthetic dataset
SYNTHE_TASK_CONFIGS = [
    {"Prompt Tokens": [17927, 25413, 4525, 5478], "name":["set1"]}, 
    {"Prompt Tokens": [29402, 7006, 17969, 10260], "name":["set2"]}, 
    {"Prompt Tokens": [16166, 7219, 18368, 21187], "name":["set3"]}, 
    {"Prompt Tokens": [21879, 18810, 4302, 12588], "name":["set4"]}, 
    {"Prompt Tokens": [15843, 29991, 2243, 12208], "name":["set5"]}, 
    {"Prompt Tokens": [2711, 25777, 11053, 8676], "name":["set6"]}, 
    {"Prompt Tokens": [8897, 14163, 9368, 6382], "name":["set7"]},
    {"Prompt Tokens": [15795, 10899, 18354, 15309], "name":["set8"]},
    {"Prompt Tokens": [8841, 12333, 21517, 3453], "name":["set9"]},
    {"Prompt Tokens": [1344, 19422, 11233, 13248], "name":["set10"]},
    {"Prompt Tokens": [21105, 11087, 23541, 2155], "name":["set11"]},
    {"Prompt Tokens": [4483, 4200, 25492, 12897], "name":["set12"]},
    {"Prompt Tokens": [4719, 9238, 28278, 27844], "name":["set13"]},
]

#The path of synthetic dataset for the "Prompt Tokens" to match.
SYNTHETIC_DATA_FILE_PATH = ""

SELECTED_MODEL = "EleutherAI/pythia-410m"  # "gpt2-medium"  EleutherAI/pythia-410m


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = "output_info"

#Choose the input cases to analyze
SELECTED_TASK_INDICES = [0]  

#For trainning dynamic analysis
USE_SYNTHETIC_DATASET = False 

#Choose the epochs to analyze
EPOCHS = [1] 
   



VISUALIZATION = True  # Global visualization switch


EAP_CONFIG = {
    "method": "EAP-IG-case",  # EAP method
    "ig_steps": 50,           # IG steps
    "target_edge_count": 300, # Target number of edges to keep (default)
    "visualize_graph": False,  # Whether to generate circuit graph visualization
}



CO_XSUB_MODE = {
    'enable': True,  # Whether to use Co method to automatically locate X_sub
    'eap': True,    # Whether to use the location result for EAP analysis (otherwise just print the matching result)
    'method': 'phantomcircuit'  # X_sub location method, options are 'phantomcircuit' or 'complex' for CoDA
}


ANALYSIS_CONFIG = {
    "analyze_circuit": True,           # Whether to analyze the logits of the final circuit
    "evaluate_performance": True,      # Whether to evaluate circuit performance
    "top_k_eval": 5,                   # top-k tokens to display during logit analysis
    "analyze_layer_outputs": True,     # Whether to analyze the output of each layer
    "analyze_component": [],          # Optional: Specify a list of components to analyze
    "max_new_tokens": 1,               # Limit the number of generated tokens to 1
    "analyze_attention_patterns": True, # Whether to analyze the attention patterns of attention heads in the circuit
    "analyze_reversed_circuit": False,   # Whether to use the reversed input
    "save_attention_plots": True,      # Whether to save attention head heatmaps, if False only metadata is kept
    "save_layer_output_heatmap": False, # If analyze_layer_outputs is True, controls whether to additionally generate and save a heatmap of word rankings for each layer's output. The saving of metadata (like raw logits) should be controlled by analyze_layer_outputs.
    "nodes_to_remove_from_circuit": [ ], # Specify a list of node names to be removed from the circuit (e.g., ["a10.h2", "m5"])
}


EDGE_OPTIMIZATION_CONFIG = {
    "enable": False,          
    "optimization_goal": "max_improvement",  # Optimization goal: max_improvement or target
    "optimization_method": "uniform_interval", # Optimization method: golden_section or uniform_interval
    "initial_edge_count": None, # Initial number of edges, None means use target_edge_count from EAP_CONFIG
    "step_size": None,          # Step size, None means auto-calculate (0.1% of total edges)
    "uniform_step_size": 100,   # Step size for uniform interval search 
    "edge_count_range": [200,5000], # Edge count range [min, max], empty list means auto-calculate
    "target_performance": None, # Target performance value, used only when optimization_goal is target
    "max_iterations": 50,
    "detailed_analysis": True,  # Whether to perform detailed circuit analysis for each edge count during optimization
}


MODEL_CONFIGS = {
   
   
    "EleutherAI/pythia-70m": {
        "path_template": "/model/pythia70m/epoch_{epoch_num}",
        "dtype": "float32",
        "trust_remote_code": False,
       
    },
    "EleutherAI/pythia-410m": {
        "path_template": "/model/pythia410m/epoch_{epoch_num}",
        "dtype": "float32",
        "trust_remote_code": False,
      
    },
  
    "EleutherAI/pythia-1.4b": {
        "path_template": "/model/pythia1.4/epoch_{epoch_num}",
        "dtype": "float32",
        "trust_remote_code": False,
      
    },
   
    "EleutherAI/pythia-2.8b": {
        "path_template": "/model/pythia2.8/epoch_{epoch_num}",
        "dtype": "float32",
        "trust_remote_code": False,
        
    },
  
    "allenai/OLMo-7B-Instruct": {
        "path_template": "/model/olmo-7b-instruct/epoch_{epoch_num}",
        "dtype": "bfloat16",
        "trust_remote_code": True,
    },

    "meta-llama/Llama-2-7b-hf": {
        "path_template": "/model/llama2-7b/epoch_{epoch_num}",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        
        
    },
    "meta-llama/Llama-2-7b-hf-reinitialized": {
        "path_template": "/model/llama2-7b-reinit/epoch_{epoch_num}",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        
    },
    "gpt2-medium": {
        "path_template": "/model/gpt/epoch_{epoch_num}",
        "dtype": "float32",
        "trust_remote_code": False,
    },
}

# Device configuration
DEVICE_CONFIG = {
    "force_device": "gpu",  
    "auto_fallback": False,  }

def parse_args():
    """Parse command line arguments to allow selecting model and task from the command line."""
    global USE_SYNTHETIC_DATASET
    global CO_XSUB_MODE
    global SYNTHE_TASK_CONFIGS
    global DEVICE_CONFIG
    parser = argparse.ArgumentParser(description="Run EAP knowledge circuit analysis")
    
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=SELECTED_MODEL,
        choices=list(MODEL_CONFIGS.keys()),
        help=f"Select the model to use, default is {SELECTED_MODEL}"
    )
    
    parser.add_argument(
        "--tasks", "-t",
        type=int,
        nargs="+",  # Allow multiple integers
        default=SELECTED_TASK_INDICES,
        help=f"Select the list of task indices to run, default is {SELECTED_TASK_INDICES}"
    )
    
    parser.add_argument(
        "--use-synthetic-dataset",
        action="store_true",
        default=USE_SYNTHETIC_DATASET, 
        help=f"Whether to use the synthetic dataset, default is {USE_SYNTHETIC_DATASET}. If True, SYNTHE_TASK_CONFIGS will be used."
    )
    
    parser.add_argument(
        "--visualization",
        action="store_true",
        default=VISUALIZATION,
        help=f"Whether to generate all visualization charts, default is {VISUALIZATION}"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default=EAP_CONFIG["method"],
        choices=["EAP-IG-case", "EAP-IG-abs", "EAP-case", "EAP-abs"],
        help=f"EAP method, default is {EAP_CONFIG['method']}"
    )
    
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=EAP_CONFIG["ig_steps"],
        help=f"Number of IG steps, default is {EAP_CONFIG['ig_steps']}"
    )
    
    parser.add_argument(
        "--target-edges",
        type=int,
        default=EAP_CONFIG["target_edge_count"],
        help=f"Target number of edges to keep, default is {EAP_CONFIG['target_edge_count']}"
    )
    
    parser.add_argument(
        "--visualize-circuit",
        action="store_true",
        default=EAP_CONFIG["visualize_graph"],
        help="Whether to generate circuit graph visualization"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory, default is {OUTPUT_DIR}, all outputs will be placed in this directory"
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=ANALYSIS_CONFIG["max_new_tokens"],
        help=f"Maximum number of new tokens to generate, default is {ANALYSIS_CONFIG['max_new_tokens']}"
    )
    
    parser.add_argument(
        "--optimize-edges",
        action="store_true",
        default=EDGE_OPTIMIZATION_CONFIG["enable"],
        help="Enable edge count optimization mode to find the best circuit performance."
    )
    parser.add_argument(
        "--optimization-goal",
        type=str,
        choices=["max_improvement", "target"],
        default=EDGE_OPTIMIZATION_CONFIG["optimization_goal"],
        help="Optimization goal for edge count: max_improvement (maximize performance improvement) or target (approach target performance value)"
    )
    parser.add_argument(
        "--initial-edges",
        type=int,
        default=EDGE_OPTIMIZATION_CONFIG["initial_edge_count"], 
        help="Starting point for edge count optimization. If not specified, the value of --target-edges is used."
    )

    parser.add_argument(
        "--step-size",
        type=int,
        default=EDGE_OPTIMIZATION_CONFIG["step_size"], 
        help="Step size for edge count optimization. If not specified, it is automatically calculated as 0.1% of the total number of edges."
    )
    parser.add_argument(
        "--edge-range",
        type=int,
        nargs='*',
        default=EDGE_OPTIMIZATION_CONFIG["edge_count_range"], 
        help="Range for edge count optimization [min_edges max_edges]. If not specified, it is automatically calculated."
    )
    parser.add_argument(
        "--target-performance",
        type=float,
        default=EDGE_OPTIMIZATION_CONFIG["target_performance"], 
        help="Target performance value, used only when --optimization-goal=target."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=EDGE_OPTIMIZATION_CONFIG["max_iterations"],
        help="Maximum number of iterations for the optimization algorithm."
    )
    
    parser.add_argument(
        "--optimization-method",
        type=str,
        choices=["golden_section", "uniform_interval"],
        default=EDGE_OPTIMIZATION_CONFIG["optimization_method"],
        help="Optimization method: golden_section (golden section search) or uniform_interval (uniform interval search)"
    )
    
    parser.add_argument(
        "--uniform-step-size",
        type=int,
        default=EDGE_OPTIMIZATION_CONFIG["uniform_step_size"],
        help="Step size for uniform interval search method (e.g., 100, 1000). Each test will increase edge count by this amount."
    )
    
    parser.add_argument(
        "--detailed-analysis",
        action="store_true",
        default=EDGE_OPTIMIZATION_CONFIG["detailed_analysis"],
        help="Enable detailed circuit analysis for each edge count during optimization (includes performance, attention metrics, high attention heads count)"
    )
    
    parser.add_argument(
        "--skip-reversed-analysis",
        action="store_false", # Set to store_false, default is True (from ANALYSIS_CONFIG)
        dest="analyze_reversed_circuit", # Store the value in args.analyze_reversed_circuit
        help="If specified, skips the full analysis of the reversed circuit (including attention patterns)."
    )
    parser.set_defaults(analyze_reversed_circuit=ANALYSIS_CONFIG["analyze_reversed_circuit"])
    
    parser.add_argument(
        "--save-attention-plots",
        action="store_true", # Note: This setup makes it always true if default is true. Consider --no-save-attention-plots with store_false.
        default=ANALYSIS_CONFIG["save_attention_plots"],
        help="If specified, saves heatmaps of attention heads. Otherwise, only the metadata of the attention analysis is kept."
    )
    parser.set_defaults(save_attention_plots=ANALYSIS_CONFIG["save_attention_plots"])

    parser.add_argument(
        "--no-save-layer-output-heatmap",
        action="store_false",
        dest="save_layer_output_heatmap",
        help="When analyze_layer_outputs is True, disables the generation and saving of layer output word ranking heatmaps. Heatmaps are generated by default. This option does not affect the analysis or saving of raw layer output data (like logits) controlled by analyze_layer_outputs."
    )
    parser.set_defaults(save_layer_output_heatmap=ANALYSIS_CONFIG["save_layer_output_heatmap"])
    
    parser.add_argument(
        "--remove-circuit-nodes",
        type=str,
        nargs="*",
        default=ANALYSIS_CONFIG["nodes_to_remove_from_circuit"],
        help="Specify a list of node names to be removed from the circuit (e.g., --remove-circuit-nodes a10.h2 m5). These nodes and their associated edges will be removed from the circuit after the EAP analysis is complete."
    )

    parser.add_argument(
        "--use-co-xsub",
        action="store_true",
        default=CO_XSUB_MODE['enable'],
        help="Whether to use the Co method to automatically locate X_sub (clean_subject) in the prompt"
    )
    parser.add_argument(
        "--co-xsub-eap",
        action="store_true",
        default=CO_XSUB_MODE['eap'],
        help="Whether to perform EAP analysis after locating X_sub with the Co method (otherwise, just print the matching results)"
    )
    parser.add_argument(
        "--co-xsub-method",
        type=str,
        default=CO_XSUB_MODE['method'],
        choices=['phantomcircuit', 'complex'],
        help="The method to use for locating X_sub with the Co method, options are 'phantomcircuit' or 'complex'"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE_CONFIG["force_device"],
        choices=[None, 'cpu', 'cuda', 'auto'],
        help="Force a specific device to be used: 'cpu', 'cuda', 'auto' (automatic detection, default), or None"
    )
    parser.add_argument(
        "--no-auto-fallback",
        action="store_false",
        dest="auto_fallback",
        help="Disable the automatic device fallback feature (does not automatically switch to another device when the specified device is unavailable)"
    )
    parser.set_defaults(auto_fallback=DEVICE_CONFIG["auto_fallback"])
    
    args = parser.parse_args()
    
    USE_SYNTHETIC_DATASET = args.use_synthetic_dataset
    CO_XSUB_MODE['enable'] = args.use_co_xsub
    CO_XSUB_MODE['eap'] = args.co_xsub_eap
    CO_XSUB_MODE['method'] = args.co_xsub_method
    
   
    EDGE_OPTIMIZATION_CONFIG["enable"] = args.optimize_edges
    EDGE_OPTIMIZATION_CONFIG["optimization_goal"] = args.optimization_goal
    EDGE_OPTIMIZATION_CONFIG["optimization_method"] = args.optimization_method
    EDGE_OPTIMIZATION_CONFIG["initial_edge_count"] = args.initial_edges
    EDGE_OPTIMIZATION_CONFIG["step_size"] = args.step_size
    EDGE_OPTIMIZATION_CONFIG["uniform_step_size"] = args.uniform_step_size
    EDGE_OPTIMIZATION_CONFIG["detailed_analysis"] = args.detailed_analysis
    if args.edge_range and len(args.edge_range) != 2:
        print(f"Error: --edge-range argument must be two values [min max], but got {len(args.edge_range)} values")
        sys.exit(1)
    EDGE_OPTIMIZATION_CONFIG["edge_count_range"] = args.edge_range
    EDGE_OPTIMIZATION_CONFIG["target_performance"] = args.target_performance
    EDGE_OPTIMIZATION_CONFIG["max_iterations"] = args.max_iterations
    
    DEVICE_CONFIG["force_device"] = args.device
    DEVICE_CONFIG["auto_fallback"] = args.auto_fallback
    
    ANALYSIS_CONFIG["analyze_reversed_circuit"] = args.analyze_reversed_circuit
    ANALYSIS_CONFIG["save_attention_plots"] = args.save_attention_plots
    ANALYSIS_CONFIG["save_layer_output_heatmap"] = args.save_layer_output_heatmap
    ANALYSIS_CONFIG["nodes_to_remove_from_circuit"] = args.remove_circuit_nodes
    
    active_task_configs = SYNTHE_TASK_CONFIGS if USE_SYNTHETIC_DATASET else TASK_CONFIGS
    valid_tasks = []
    for task_idx in args.tasks:
        if 0 <= task_idx < len(active_task_configs):
            valid_tasks.append(task_idx)
        else:
            print(f"Warning: Task index {task_idx} is out of range (0-{len(active_task_configs)-1}) and will be ignored")
    
    if not valid_tasks:
        print(f"Error: No valid task indices provided! Using default values: {SELECTED_TASK_INDICES} (relative to the selected dataset)")
        valid_tasks = [idx for idx in SELECTED_TASK_INDICES if 0 <= idx < len(active_task_configs)]
        if not valid_tasks and len(active_task_configs) > 0 : 
             print(f"Warning: Default task indices are also invalid, attempting to use task 0")
             valid_tasks = [0] if len(active_task_configs) > 0 else []
        elif not valid_tasks and len(active_task_configs) == 0:
            print(f"Error: Both the selected and default task lists are invalid because the task configuration list is empty ({'SYNTHE_TASK_CONFIGS' if USE_SYNTHETIC_DATASET else 'TASK_CONFIGS'} is empty)")

    args.tasks = valid_tasks
    

    return args 