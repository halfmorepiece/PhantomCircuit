import torch
import matplotlib.pyplot as plt
import seaborn as sns
import transformer_lens.utils as utils
from rich import print as rprint
import os
import traceback
import numpy as np
import json

def get_component_logits_local(logits, model, answer_token, top_k=5, compare_with_full_model=False, input_tokens=None, max_new_tokens=1, prefix=""):
    """
    Analyze and print the logits output of a single component (or full circuit).
    
    Args:
        logits: Output logits tensor (usually shape [seq_len, d_vocab])
        model: HookedTransformer model instance
        answer_token: Integer ID of the target correct answer token
        top_k: Number of top tokens to display
        compare_with_full_model: If True, also display the prediction of the full model (requires input_tokens)
        input_tokens: If compare_with_full_model=True, then input tokens are required for the full model
        max_new_tokens: Maximum number of new tokens to generate (No longer used for printing)
        prefix: String to add before output information (e.g., "Modified Circuit")
        
    Returns:
        dict: Dictionary containing rank and logit information, including:
            - circuit_rank: Rank of token in the circuit
            - circuit_logit: Logit value of token in the circuit
            - circuit_prob: Probability of token in the circuit
            - circuit_top_tokens: List of top K tokens in the circuit
            - full_model_rank: Rank of token in the full model (if compare_with_full_model=True)
            - full_model_logit: Logit value of token in the full model (if compare_with_full_model=True)
            - full_model_prob: Probability of token in the full model (if compare_with_full_model=True)
            - full_model_top_tokens: List of top K tokens in the full model (if compare_with_full_model=True)
    """
    # Initialize return result dictionary
    result = {
        "circuit_rank": -1,
        "circuit_logit": None,
        "circuit_prob": None,
        "circuit_token_str": None,
        "circuit_top_tokens": [],
        "full_model_rank": -1,
        "full_model_logit": None,
        "full_model_prob": None,
        "full_model_token_str": None,
        "full_model_top_tokens": []
    }
    
    if logits is None:
        print("Error: Received None logits in get_component_logits_local.")
        return result
        
    if not isinstance(logits, torch.Tensor):
        print(f"Error: Expected logits to be a torch.Tensor, got {type(logits)}.")
        return result
        
    if logits.ndim == 3 and logits.shape[0] == 1:  # Handle possible batch dimension
        logits = logits.squeeze(0)
    elif logits.ndim != 2:
        print(f"Error: Expected logits to be 2D (seq_len, d_vocab), got shape {logits.shape}.")
        return result

    # Ensure answer_token is a valid integer ID
    try:
        answer_token_int = int(answer_token)
        if not (0 <= answer_token_int < model.cfg.d_vocab):
            raise ValueError(f"answer_token {answer_token_int} is out of vocab size {model.cfg.d_vocab}")
    except (ValueError, TypeError) as e:
        print(f"Error: Invalid answer_token: {answer_token}. Needs to be a valid integer token ID. {e}")
        return result

    try:
        prefix_str = f"{prefix}: " if prefix else ""
        # We usually care about the prediction at the last token position
        final_logits = logits[-1, :]  # Shape: [d_vocab]
        probs = final_logits.softmax(dim=-1)

        # Safely get the string representation of the answer token
        try:
            answer_str_token = model.to_string([answer_token_int])  # Passed as list
            if isinstance(answer_str_token, list):  # Handle possible list output
                answer_str_token = answer_str_token[0]
        except Exception as e_str:
            # print(f"Warning: Could not convert answer token ID {answer_token_int} to string: {e_str}") # Removed
            answer_str_token = f"[ID:{answer_token_int}]"

        sorted_token_probs, sorted_token_indices = probs.sort(descending=True)

        # Find the rank of the correct answer token
        sorted_indices_list = sorted_token_indices.cpu().tolist()
        try:
            # Find the first occurrence of the answer token ID
            correct_rank = sorted_indices_list.index(answer_token_int)
        except ValueError:
            correct_rank = -1  # Indicates not found

        # Save information to result dictionary
        result["circuit_rank"] = correct_rank
        result["circuit_logit"] = final_logits[answer_token_int].item() if correct_rank != -1 else None
        result["circuit_prob"] = probs[answer_token_int].item() if correct_rank != -1 else None
        result["circuit_token_str"] = answer_str_token

        # Print to terminal
        rprint(
            f"{prefix_str}Performance on answer token:\n[b]Rank: {correct_rank if correct_rank != -1 else 'Not Found': <8} Logit: {final_logits[answer_token_int].item():5.2f} Prob: {probs[answer_token_int].item():6.2%} Token: |{answer_str_token}|[/b]"
        )

        print(f"\n{prefix_str}Top {top_k} predictions:") # Adjusted print
        
        # Save top K prediction token information
        circuit_top_tokens = []
        for i in range(min(top_k, len(sorted_indices_list))):
            token_id = sorted_token_indices[i].item()
            try:
                token_str = model.to_string([token_id])
                if isinstance(token_str, list):
                    token_str = token_str[0]
            except Exception as e_str_top:
                # print(f"Warning: Could not convert predicted token ID {token_id} to string: {e_str_top}") # Removed
                token_str = f"[ID:{token_id}]"

            # Save to result dictionary
            circuit_top_tokens.append({
                "rank": i,
                "token_id": token_id,
                "token_str": token_str,
                "logit": final_logits[token_id].item(),
                "prob": sorted_token_probs[i].item()
            })

            print(
                f"  Rank {i}: Logit: {final_logits[token_id].item():5.2f} Prob: {sorted_token_probs[i].item():6.2%} Token: |{token_str}|"
            )
            
        result["circuit_top_tokens"] = circuit_top_tokens

        # If comparison with full model output is needed
        if compare_with_full_model:
            if input_tokens is None:
                print(f"Error: {prefix_str}input_tokens required when compare_with_full_model=True.") # Added prefix
                return result

            print(f"\n{prefix_str}-------- Full Model Comparison --------") # Added prefix
            with torch.no_grad():
                # Ensure input_tokens is tensor form
                if isinstance(input_tokens, str):
                    input_tokens = model.to_tokens(input_tokens)
                elif not isinstance(input_tokens, torch.Tensor):
                    print(f"Error: {prefix_str}Unexpected input_tokens type: {type(input_tokens)}") # Added prefix
                    return result

                # Run full model to get prediction
                full_model_logits = model(input_tokens)
                full_model_final_logits = full_model_logits[0, -1, :]  # Take logits at last position
                full_model_probs = torch.softmax(full_model_final_logits, dim=-1)

                # Calculate rank
                full_model_sorted_probs, full_model_sorted_indices = full_model_probs.sort(descending=True)
                full_model_sorted_indices_list = full_model_sorted_indices.cpu().tolist()

                # Find the rank of the target token
                try:
                    full_model_correct_rank = full_model_sorted_indices_list.index(answer_token_int)
                except ValueError:
                    full_model_correct_rank = -1

                # Save full model information to result dictionary
                result["full_model_rank"] = full_model_correct_rank
                result["full_model_logit"] = full_model_final_logits[answer_token_int].item() if full_model_correct_rank != -1 else None
                result["full_model_prob"] = full_model_probs[answer_token_int].item() if full_model_correct_rank != -1 else None
                result["full_model_token_str"] = answer_str_token

                # Print target token performance
                rprint(
                    f"{prefix_str}Full model performance on answer token:\n[b]Rank: {full_model_correct_rank if full_model_correct_rank != -1 else 'Not Found': <8} Logit: {full_model_final_logits[answer_token_int].item():5.2f} Prob: {full_model_probs[answer_token_int].item():6.2%} Token: |{answer_str_token}|[/b]"
                )

                # Save top K predictions of full model
                full_model_top_tokens = []
                print(f"\n{prefix_str}Top {top_k} predictions (Full model):") # Added prefix
                for i in range(min(top_k, len(full_model_sorted_indices_list))):
                    token_id = full_model_sorted_indices[i].item()
                    try:
                        token_str = model.to_string([token_id])
                        if isinstance(token_str, list):
                            token_str = token_str[0]
                    except Exception as e_str_top:
                        # print(f"Warning: Could not convert predicted token ID {token_id} to string: {e_str_top}") # Removed
                        token_str = f"[ID:{token_id}]"

                    # Save to result dictionary
                    full_model_top_tokens.append({
                        "rank": i,
                        "token_id": token_id,
                        "token_str": token_str,
                        "logit": full_model_final_logits[token_id].item(),
                        "prob": full_model_sorted_probs[i].item()
                    })

                    print(
                        f"  Rank {i}: Logit: {full_model_final_logits[token_id].item():5.2f} Prob: {full_model_sorted_probs[i].item():6.2%} Token: |{token_str}|"
                    )
                    
                result["full_model_top_tokens"] = full_model_top_tokens

    except IndexError as e:
        print(f"Error accessing logits or tokens ({prefix_str}IndexError): {e}. Logits shape: {logits.shape}") # Added prefix
    except Exception as e:
        print(f"An unexpected error occurred in {prefix_str}get_component_logits_local: {e}") # Added prefix
        traceback.print_exc()
        
    return result

def draw_attention_pattern(cache, tokens, model, layer, head_index, 
                      layer_top_k_tokens=None, layer_top_k_probs=None, layer_top_k_logits=None,
                      filename="attention_pattern.png"):
    """Draw attention pattern visualization for a specific attention head and save
    Args:
        cache: Activation cache from model run
        tokens: Input token IDs
        model: HookedTransformer model
        layer: Layer index
        head_index: Attention head index
        layer_top_k_tokens: Optional top-k token string list
        layer_top_k_probs: Optional top-k probability list
        layer_top_k_logits: Optional top-k logits list
        filename: Output file name
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

        # Create token labels, use actual characters
        token_labels = token_strs

        # Whether to draw combined figure (includes attention and top layer logits)
        if layer_top_k_tokens and layer_top_k_probs and layer_top_k_logits:
            # Draw combined figure (attention pattern + top logits)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=150, gridspec_kw={'width_ratios': [2, 1]})
            
            # Left: Attention pattern heatmap
            sns.heatmap(attention_pattern, cmap='viridis',
                        xticklabels=token_labels, yticklabels=token_labels, 
                        annot=False, cbar=True, ax=ax1, square=True)
            ax1.set_title(f'Attention Pattern - Head L{layer}.H{head_index}')
            ax1.set_xlabel('Keys')
            ax1.set_ylabel('Queries')
            ax1.tick_params(axis='x', rotation=90, labelsize=8)
            ax1.tick_params(axis='y', labelsize=8)
            
            # Right: Logits rank bar chart
            y_pos = np.arange(len(layer_top_k_tokens))
            bars = ax2.barh(y_pos, layer_top_k_probs, align='center')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(layer_top_k_tokens)
            ax2.invert_yaxis()  # Highest probability at top
            ax2.set_xlabel('Probability')
            ax2.set_title(f'Top Predictions from Layer {layer}')
            
            # Add probability and logit value labels
            for i, (bar, prob, logit) in enumerate(zip(bars, layer_top_k_probs, layer_top_k_logits)):
                ax2.text(max(0.01, prob/2), bar.get_y() + bar.get_height()/2, 
                        f"{prob:.2%}\nL:{logit:.2f}", 
                        ha='center', va='center', fontsize=8, color='white' if prob > 0.3 else 'black')
                
            fig.tight_layout()
            plt.savefig(filename)
            plt.close(fig)
        else:
            # Draw basic attention pattern figure
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
            plt.close()

        print(f"Attention pattern saved to {filename}")
    except Exception as e:
        print(f"Error generating attention pattern visualization: {e}")
        import traceback
        traceback.print_exc()

def analyze_layer_outputs(model, clean_tokens, label_tensor, output_dir, task_name, target_edge_count, ig_steps, input_prompt=None):
    """
    Analyze model output Logits and ranks for each layer.
    
    Args:
        model: HookedTransformer model instance
        clean_tokens: Clean text token tensor 
        label_tensor: Tensor containing correct and incorrect answer tokens
        output_dir: Output directory path
        task_name: Task name
        target_edge_count: Edge count for file naming
        ig_steps: IG step for file naming
        input_prompt: Input prompt text (optional)
    """
    try:
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
                    raise ValueError(f"Cannot handle type {type(clean_tokens)} of clean_tokens")
            
            # Use run_with_cache to get all activations
            original_logits, cache = model.run_with_cache(clean_tokens)

            num_layers = model.cfg.n_layers
            layer_indices = list(range(num_layers))
            correct_logits_per_layer = []
            incorrect_logits_per_layer = []
            correct_ranks_per_layer = []
            incorrect_ranks_per_layer = []
            
            # New: Save top k vocabulary and logits for each layer
            top_k = 10  # Top k words to save for each layer
            layer_top_tokens = []  # Top k vocabulary IDs for each layer
            layer_top_token_strs = []  # Top k vocabulary text for each layer
            layer_top_logits = []  # Top k logits values for each layer
            layer_top_probs = []  # Top k probability values for each layer

            # Get correct and incorrect answer token IDs and string representations
            correct_idx = label_tensor[0, 0].item()
            incorrect_idx = label_tensor[0, 1].item()
            
            try:
                correct_token_str = model.to_string([correct_idx])
                incorrect_token_str = model.to_string([incorrect_idx])
            except Exception as e:
                correct_token_str = f"[ID:{correct_idx}]"
                incorrect_token_str = f"[ID:{incorrect_idx}]"

            # Get input prompt text
            if input_prompt is None:
                try:
                    input_prompt = model.to_string(clean_tokens[0])
                except Exception as e:
                    input_prompt = f"[Input shape: {clean_tokens.shape}]"

            for layer_idx in layer_indices:
                # Get residual stream output after this layer block
                layer_resid_full = cache[f'blocks.{layer_idx}.hook_resid_post'] # Shape: [batch, seq_len, d_model]

                # Apply final LayerNorm and convert to Logits via unembedding matrix
                layer_logits_full = model.unembed(model.ln_final(layer_resid_full)) # Shape: [batch, seq_len, d_vocab]

                # Extract logits from the last token
                layer_logits = layer_logits_full[0, -1, :] # Shape: [d_vocab]

                # Calculate probability and rank
                layer_probs = torch.softmax(layer_logits, dim=-1)
                sorted_indices = torch.argsort(layer_probs, descending=True).cpu().tolist()
                
                # Get top k token IDs
                top_k_indices = sorted_indices[:top_k]
                top_k_logits = [layer_logits[idx].item() for idx in top_k_indices]
                top_k_probs = [layer_probs[idx].item() for idx in top_k_indices]
                
                # Get top k token text
                try:
                    top_k_token_strs = [model.to_string([idx]).strip() for idx in top_k_indices]
                except Exception as e:
                    top_k_token_strs = [f"[ID:{idx}]" for idx in top_k_indices]
                
                # Save data
                layer_top_tokens.append(top_k_indices)
                layer_top_token_strs.append(top_k_token_strs)
                layer_top_logits.append(top_k_logits)
                layer_top_probs.append(top_k_probs)

                # Get correct and incorrect answer Logit
                correct_logit = layer_logits[correct_idx].item()
                incorrect_logit = layer_logits[incorrect_idx].item()
                correct_logits_per_layer.append(correct_logit)
                incorrect_logits_per_layer.append(incorrect_logit)

                # Get correct and incorrect answer rank
                try:
                    correct_rank = sorted_indices.index(correct_idx) + 1
                except ValueError:
                    correct_rank = -1 # Not found (shouldn't happen for vocab IDs)
                try:
                    incorrect_rank = sorted_indices.index(incorrect_idx) + 1
                except ValueError:
                    incorrect_rank = -1
                correct_ranks_per_layer.append(correct_rank)
                incorrect_ranks_per_layer.append(incorrect_rank)

            # Create plot save path, include task name to distinguish clean/corrupted
            plot_save_path_base = os.path.join(output_dir, f"{task_name}_layer_analysis_top{target_edge_count}_ig{ig_steps}")
            
            # Create layer_outputs directory
            layer_outputs_dir = os.path.join(output_dir, "layer_outputs")
            os.makedirs(layer_outputs_dir, exist_ok=True)

            # Draw Logits figure
            plt.figure(figsize=(12, 6))
            plt.plot(layer_indices, correct_logits_per_layer, marker='o', label=f"Correct Answer: '{correct_token_str}'")
            plt.plot(layer_indices, incorrect_logits_per_layer, marker='x', label=f"Incorrect Answer: '{incorrect_token_str}'")
            plt.xlabel("Layer Index")
            plt.ylabel("Logit Value")
            plt.title(f"Logit Value of Target Answers at Each Layer Output\\nPrompt: {input_prompt}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            logits_plot_filename = f"{plot_save_path_base}_logits.png"
            try:
                plt.savefig(logits_plot_filename)
            except Exception as e_plt:
                print(f"Error saving logits plot: {e_plt}")
                pass
            plt.close()

            # Draw Ranks figure
            plt.figure(figsize=(12, 6))
            plt.plot(layer_indices, correct_ranks_per_layer, marker='o', label=f"Correct Answer: '{correct_token_str}'")
            plt.plot(layer_indices, incorrect_ranks_per_layer, marker='x', label=f"Incorrect Answer: '{incorrect_token_str}'")
            plt.xlabel("Layer Index")
            plt.ylabel("Rank (Lower is better)")
            plt.title(f"Rank of Target Answers at Each Layer Output\\nPrompt: {input_prompt}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            ranks_plot_filename = f"{plot_save_path_base}_ranks.png"
            try:
                plt.savefig(ranks_plot_filename)
            except Exception as e_plt:
                print(f"Error saving ranks plot: {e_plt}")
                pass
            plt.close()
            
            # New: Generate and save layer output word rank heatmap
            from config import ANALYSIS_CONFIG
            if ANALYSIS_CONFIG.get("save_layer_output_heatmap", True):
                # Prepare heatmap data
                # Merge top tokens of each layer into a single unique token set
                all_unique_tokens = set()
                for layer_tokens in layer_top_tokens:
                    all_unique_tokens.update(layer_tokens)
                
                # Add correct and incorrect answer tokens
                all_unique_tokens.add(correct_idx)
                all_unique_tokens.add(incorrect_idx)
                all_unique_tokens = list(all_unique_tokens)
                
                # To better visualize, we sort tokens by frequency of appearance
                token_freq = {}
                for token in all_unique_tokens:
                    freq = 0
                    for layer_tokens in layer_top_tokens:
                        if token in layer_tokens:
                            freq += 1
                    token_freq[token] = freq
                
                # Sort by frequency in descending order
                all_unique_tokens = sorted(all_unique_tokens, key=lambda x: token_freq[x], reverse=True)
                
                # Limit displayed token count to avoid large chart
                max_tokens_to_show = min(30, len(all_unique_tokens))
                tokens_to_show = all_unique_tokens[:max_tokens_to_show]
                
                # Get string representations of these tokens
                try:
                    token_strs = [model.to_string([idx]).strip() for idx in tokens_to_show]
                except Exception as e:
                    token_strs = [f"[ID:{idx}]" for idx in tokens_to_show]
                
                # Create rank matrix for each token in each layer
                rank_matrix = np.zeros((len(tokens_to_show), num_layers))
                for i, token_id in enumerate(tokens_to_show):
                    for j, layer_idx in enumerate(layer_indices):
                        # Get sort index of this layer
                        layer_sorted_indices = torch.argsort(torch.softmax(layer_logits_full[0, -1, :], dim=-1), descending=True).cpu().tolist()
                        try:
                            rank = layer_sorted_indices.index(token_id) + 1
                        except ValueError:
                            rank = max(1000, len(layer_sorted_indices))  # Not found, give a large rank
                        rank_matrix[i, j] = min(rank, 100)  # Limit max display rank to 100
                
                # Create heatmap
                plt.figure(figsize=(15, max(10, len(tokens_to_show) * 0.3)))
                
                # Use log scale to make colors more distinguishable, scale rank_matrix
                rank_matrix_log = np.log10(rank_matrix + 1)  # +1 to avoid log(0)
                
                # Draw heatmap
                sns.heatmap(rank_matrix_log, annot=rank_matrix.astype(int), fmt="d", 
                           cmap="YlGnBu_r", linewidths=0.5, 
                           xticklabels=layer_indices, 
                           yticklabels=token_strs, 
                           cbar_kws={'label': 'Log10(Rank)'})
                
                # Add labels and title
                plt.xlabel("Layer Index")
                plt.ylabel("Token")
                plt.title(f"Token Ranking Heatmap Across Layers\nPrompt: {input_prompt[:50]}{'...' if len(input_prompt) > 50 else ''}")
                
                # Optimize chart layout
                plt.tight_layout()
                
                # Save heatmap
                heatmap_filename = os.path.join(layer_outputs_dir, f"{task_name}_layer_output_heatmap.png")
                try:
                    plt.savefig(heatmap_filename)
                    print(f"Layer output heatmap saved to {heatmap_filename}")
                except Exception as e_heatmap:
                    print(f"Error saving layer output heatmap: {e_heatmap}")
                plt.close()
            
            # Save layer output metadata
            metadata = {
                "task_name": task_name,
                "input_prompt": input_prompt,
                "correct_token": {
                    "id": correct_idx,
                    "string": correct_token_str,
                    "logits_per_layer": correct_logits_per_layer,
                    "ranks_per_layer": correct_ranks_per_layer
                },
                "incorrect_token": {
                    "id": incorrect_idx,
                    "string": incorrect_token_str,
                    "logits_per_layer": incorrect_logits_per_layer,
                    "ranks_per_layer": incorrect_ranks_per_layer
                },
                "layer_data": [
                    {
                        "layer_idx": idx,
                        "top_tokens": {
                            "ids": layer_top_tokens[idx],
                            "strings": layer_top_token_strs[idx],
                            "logits": layer_top_logits[idx],
                            "probs": layer_top_probs[idx]
                        }
                    }
                    for idx in layer_indices
                ]
            }
            
            # Save metadata
            metadata_filename = os.path.join(layer_outputs_dir, f"{task_name}_layer_output_metadata.json")
            try:
                with open(metadata_filename, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                print(f"Layer output metadata saved to {metadata_filename}")
            except Exception as e_meta:
                print(f"Error saving layer output metadata: {e_meta}")

    except Exception as e_layer_analysis:
        print(f"\nError during layer analysis: {e_layer_analysis}")
        traceback.print_exc()
        pass # Fail silently

def analyze_specific_components(model, clean_tokens, label_tensor=None, output_dir=None, task_name=None, 
                           target_edge_count=None, ig_steps=None, component_list=None, cache=None):
    """
    Analyze attention pattern and Logits output of specific components.
    
    Args:
        model: HookedTransformer model instance
        clean_tokens: Clean text token tensor 
        label_tensor: Optional, tensor containing correct and incorrect answer tokens
        output_dir: Output directory path
        task_name: Task name
        target_edge_count: Edge count for file naming
        ig_steps: IG step for file naming
        component_list: List of components to analyze
        cache: Optional, existing model cache, if None, regenerate
    """
    # Compatibility handling:
    # main.py: model, clean_tokens, label_tensor, output_dir, task_name, target_edge_count, ig_steps, component_list
    # eap_utils: model, clean_tokens, component_list, cache, output_dir, task_name, target_edge_count, ig_steps, draw_func
    
    # Handle case called from eap_utils.py, in which case component_list parameter is in 3rd position
    if label_tensor is not None and isinstance(label_tensor, list):
        component_list = label_tensor
        label_tensor = None
        # Handle case where target_edge_count is in 7th position
        if output_dir is not None and isinstance(output_dir, (int, float)):
            target_edge_count = output_dir
        
    # Ensure component_list is valid
    if component_list is None or not isinstance(component_list, list) or len(component_list) == 0:
        print("Warning: No valid components provided for analysis.")
        return
        
    try:
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
                    raise ValueError(f"Cannot handle type {type(clean_tokens)} of clean_tokens")
            
            # If no cache provided, use run_with_cache to generate
            if cache is None:
                original_logits, cache = model.run_with_cache(clean_tokens)

            for component_name in component_list:
                # Only process attention heads ('a'开头, 包含'.h')
                if component_name.startswith('a') and '.h' in component_name:
                    try:
                        layer_str, head_str = component_name.split('.')
                        layer = int(layer_str[1:])  # Remove 'a' prefix
                        head = int(head_str[1:])    # Remove 'h' prefix
                        
                        # --- Analyze Logits output of this layer (first calculate, then draw figure needed) ---
                        layer_top_k_tokens = []
                        layer_top_k_probs = []
                        layer_top_k_logits = []
                        print(f"\n--- Analyzing Layer {layer} Output Logits (associated with {component_name}) ---")
                        resid_post_key = f'blocks.{layer}.hook_resid_post'
                        if resid_post_key in cache:
                            layer_resid_post = cache[resid_post_key] # Shape: [batch, seq_len, d_model]
                            layer_logits_full = model.unembed(model.ln_final(layer_resid_post)) # Shape: [batch, seq_len, d_vocab]
                            layer_final_logits = layer_logits_full[0, -1, :] # Shape: [d_vocab]
                            layer_probs = torch.softmax(layer_final_logits, dim=-1)
                            top_k_layer = 5
                            top_probs, top_indices = torch.topk(layer_probs, k=top_k_layer)
                            
                            print(f"Top {top_k_layer} predictions from Layer {layer} output:")
                            for i in range(top_k_layer):
                                token_id = top_indices[i].item()
                                try:
                                    token_str = model.to_string([token_id])
                                    if isinstance(token_str, list): token_str = token_str[0]
                                except Exception: token_str = f"[ID:{token_id}]"
                                current_logit = layer_final_logits[token_id].item()
                                current_prob = top_probs[i].item()
                                layer_top_k_tokens.append(token_str)
                                layer_top_k_logits.append(current_logit)
                                layer_top_k_probs.append(current_prob)
                                print(f"  Rank {i+1}: Logit: {current_logit:5.2f} Prob: {current_prob:6.2%} Token: |{token_str}|")
                        else:
                            print(f"Warning: Residual stream output key '{resid_post_key}' not found in cache.")
                        # --- Logits analysis ended ---

                        # --- Draw combined figure --- 
                        # Use appropriate formatting based on type (float percentage or int count)
                        if isinstance(target_edge_count, float):
                            debug_suffix = f"_debug_top{target_edge_count:.2f}pct_ig{ig_steps}"
                        elif isinstance(target_edge_count, int):
                            debug_suffix = f"_debug_edges{target_edge_count}_ig{ig_steps}"
                        else:
                            debug_suffix = f"_debug_unknown{target_edge_count}_ig{ig_steps}" # Fallback
                            
                        combined_plot_filename = os.path.join(output_dir, f"{component_name}_attn_layer_logits{debug_suffix}.png")
                        draw_attention_pattern(
                            cache, clean_tokens, model, layer, head, 
                            layer_top_k_tokens, layer_top_k_probs, layer_top_k_logits, # Pass calculated logits data
                            filename=combined_plot_filename
                        )
                            
                    except ValueError:
                        print(f"Warning: Could not parse layer/head from component name: {component_name}")
                        pass
                    except Exception as e_comp:
                         print(f"Error analyzing component {component_name}: {e_comp}")
                         traceback.print_exc() # Keep traceback here
                         pass
                else:
                    pass

    except Exception as e_analyze_specific:
        print(f"\nError during specific component analysis: {e_analyze_specific}")
        traceback.print_exc()
        pass

def analyze_attention_for_subjects(model, g, clean_tokens, corrupted_tokens, task_config, output_dir, task_name, is_synthetic=False, current_epoch=None, save_plots=True):
    """
    Analyze attention patterns of attention heads in the circuit towards differential parts (subjects) in the input.
    
    Args:
        model: HookedTransformer model instance
        g: Circuit graph object
        clean_tokens: Token sequence of clean text
        corrupted_tokens: Token sequence of corrupted text
        task_config: Task configuration
        output_dir: Output directory
        task_name: Task name
        is_synthetic: Whether it's synthetic data
        current_epoch: Current processing epoch (new parameter)
        save_plots: Whether to save attention pattern plots as images

    Returns:
        dict: Dictionary containing attention analysis metrics
    """
    # Set matplotlib to use basic fonts
    import matplotlib
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    
    if g is None:
        print("Error: Graph object is None, cannot analyze attention patterns.")
        return {}

    # Ensure tokens are tensors and have batch dimension
    if isinstance(clean_tokens, torch.Tensor):
        if clean_tokens.dim() == 1:
            clean_tokens = clean_tokens.unsqueeze(0)
    else:
        print(f"Error: clean_tokens is not a tensor (type: {type(clean_tokens)})")
        return {}
        
    if isinstance(corrupted_tokens, torch.Tensor):
        if corrupted_tokens.dim() == 1:
            corrupted_tokens = corrupted_tokens.unsqueeze(0)
    else:
        print(f"Error: corrupted_tokens is not a tensor (type: {type(corrupted_tokens)})")
        return {}

    # Add epoch information to output
    epoch_str = f"_epoch{current_epoch}" if current_epoch is not None else ""
    print(f"\nAnalyzing attention patterns{epoch_str}...")

    # Preprocess input tokens, remove padding and special symbols
    # Identify padding and special symbols
    special_tokens = {'<endoftext>', '<|endoftext|>', '<s>', '</s>', '<pad>', '[PAD]'}
    padding_token_ids = []
    
    # Try to get padding token ID from tokenizer
    if model.tokenizer:
        for attr in ['pad_token_id', 'eos_token_id', 'bos_token_id']:
            if hasattr(model.tokenizer, attr) and getattr(model.tokenizer, attr) is not None:
                padding_token_ids.append(getattr(model.tokenizer, attr))
    
    # Convert tokens to string to identify special symbols
    clean_tokens_str = model.to_str_tokens(clean_tokens[0])
    corrupted_tokens_str = model.to_str_tokens(corrupted_tokens[0])
    
    # Find valid positions without padding/special symbols
    clean_valid_indices = []
    corrupted_valid_indices = []
    
    # Check clean tokens
    for i, (token_id, token_str) in enumerate(zip(clean_tokens[0].cpu().tolist(), clean_tokens_str)):
        if token_id not in padding_token_ids and token_str.strip() not in special_tokens:
            clean_valid_indices.append(i)
    
    # Check corrupted tokens
    for i, (token_id, token_str) in enumerate(zip(corrupted_tokens[0].cpu().tolist(), corrupted_tokens_str)):
        if token_id not in padding_token_ids and token_str.strip() not in special_tokens:
            corrupted_valid_indices.append(i)
    
    print(f"Clean tokens: Original length={len(clean_tokens_str)}, Valid indices={clean_valid_indices}, Valid length={len(clean_valid_indices)}")
    print(f"Corrupted tokens: Original length={len(corrupted_tokens_str)}, Valid indices={corrupted_valid_indices}, Valid length={len(corrupted_valid_indices)}")
    
    # If no valid indices, cannot continue analysis
    if not clean_valid_indices or not corrupted_valid_indices:
        print("Warning: No valid non-padding tokens found, cannot perform attention analysis")
        return {}
    
    # Create new tensor with only valid tokens
    clean_tokens_filtered = clean_tokens.clone()
    corrupted_tokens_filtered = corrupted_tokens.clone()
    
    # For cases where sequence length needs to be adjusted, we create a new tensor, only with valid tokens
    if len(clean_valid_indices) < clean_tokens.shape[1]:
        # Create a new tensor of appropriate size
        device = clean_tokens.device
        clean_tokens_new = torch.zeros((1, len(clean_valid_indices)), dtype=clean_tokens.dtype, device=device)
        for new_idx, orig_idx in enumerate(clean_valid_indices):
            clean_tokens_new[0, new_idx] = clean_tokens[0, orig_idx]
        clean_tokens_filtered = clean_tokens_new
        print(f"New clean tensor with only valid tokens created, new shape={clean_tokens_filtered.shape}")
    
    if len(corrupted_valid_indices) < corrupted_tokens.shape[1]:
        # Create a new tensor of appropriate size
        device = corrupted_tokens.device
        corrupted_tokens_new = torch.zeros((1, len(corrupted_valid_indices)), dtype=corrupted_tokens.dtype, device=device)
        for new_idx, orig_idx in enumerate(corrupted_valid_indices):
            corrupted_tokens_new[0, new_idx] = corrupted_tokens[0, orig_idx]
        corrupted_tokens_filtered = corrupted_tokens_new
        print(f"New corrupted tensor with only valid tokens created, new shape={corrupted_tokens_filtered.shape}")
    
    # Use filtered tokens to run model and get cache
    try:
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens_filtered)
            _, corrupted_cache = model.run_with_cache(corrupted_tokens_filtered)
    except Exception as e:
        print(f"Error running model to get cache with filtered tokens: {e}")
        traceback.print_exc()
        # If filtered tokens run fails, fall back to using original tokens
        print("Falling back to using original tokens to run model...")
        try:
            with torch.no_grad():
                _, clean_cache = model.run_with_cache(clean_tokens)
                _, corrupted_cache = model.run_with_cache(corrupted_tokens)
        except Exception as e2:
            print(f"Error running model to get cache with original tokens: {e2}")
            traceback.print_exc()
            return {}
    
    # Update tokens and string representations
    clean_tokens = clean_tokens_filtered
    corrupted_tokens = corrupted_tokens_filtered
    clean_tokens_str = model.to_str_tokens(clean_tokens[0])
    corrupted_tokens_str = model.to_str_tokens(corrupted_tokens[0])
    
    # All tokens are now valid, clear padding indices
    clean_pad_indices = []
    corrupted_pad_indices = []

    # Find all attention head nodes in the circuit
    attention_nodes = []
    for node_name in g.nodes:
        # Only analyze attention heads remaining in the pruned circuit
        if node_name.startswith('a') and '.h' in node_name and g.nodes[node_name].in_graph:
            # Verify that the node is actually in the graph
            if hasattr(g.nodes[node_name], 'included') and g.nodes[node_name].included:
                # Typical attention head node name format is "aL.hH", where L is the layer number and H is the head number
                attention_nodes.append(node_name)
            elif not hasattr(g.nodes[node_name], 'included'):
                # If the node doesn't have the 'included' attribute but in_graph is True, include it as well
                attention_nodes.append(node_name)
    
    if not attention_nodes:
        print("No attention head nodes found in the circuit.")
        return {}
    
    print(f"Found {len(attention_nodes)} attention head nodes in the circuit: {attention_nodes}")
    
    # Convert tokens to IDs and strings for debugging and visualization
    clean_tokens_list = clean_tokens[0].cpu().tolist()
    corrupted_tokens_list = corrupted_tokens[0].cpu().tolist()

    # BOS has already been handled, so remove the original handling logic
    # --- BOS Token Handling ---
    clean_bos_present = False
    if model.tokenizer and hasattr(model.tokenizer, 'bos_token_id') and model.tokenizer.bos_token_id is not None:
        if clean_tokens.ndim > 1 and clean_tokens.shape[1] > 0 and clean_tokens[0, 0] == model.tokenizer.bos_token_id:
            clean_bos_present = True
            if len(clean_tokens_list) > 0: clean_tokens_list = clean_tokens_list[1:]
            if len(clean_tokens_str) > 0: clean_tokens_str = clean_tokens_str[1:]

    corrupted_bos_present = False
    if model.tokenizer and hasattr(model.tokenizer, 'bos_token_id') and model.tokenizer.bos_token_id is not None:
        if corrupted_tokens.ndim > 1 and corrupted_tokens.shape[1] > 0 and corrupted_tokens[0, 0].item() == model.tokenizer.bos_token_id:
            corrupted_bos_present = True
            if len(corrupted_tokens_list) > 0: corrupted_tokens_list = corrupted_tokens_list[1:]
            if len(corrupted_tokens_str) > 0: corrupted_tokens_str = corrupted_tokens_str[1:]
    
    # Print token sequence for debugging
    print("\n=== Clean Tokens Sequence (after filtering) ===")
    for i, (token_id, token) in enumerate(zip(clean_tokens_list, clean_tokens_str)):
        print(f"{i}: '{token}' (ID: {token_id})")
    
    print("\n=== Corrupted Tokens Sequence (after filtering) ===")
    for i, (token_id, token) in enumerate(zip(corrupted_tokens_list, corrupted_tokens_str)):
        print(f"{i}: '{token}' (ID: {token_id})")
    
    # Get the distinct parts (subjects) in the input, excluding padding tokens
    clean_subject_indices = []
    corrupted_subject_indices = []
    
    # Handle different types of task configurations (synthetic or regular data)
    if is_synthetic:
        print("\nDetecting distinct parts using synthetic dataset...")
        
        # Find different parts in tokens
        # Since we've already removed padding tokens, we can directly compare
        min_len = min(len(clean_tokens_list), len(corrupted_tokens_list))
        
        # Scan from front to back to find the first different position
        first_diff = None
        for i in range(min_len):
            if clean_tokens_list[i] != corrupted_tokens_list[i]:
                first_diff = i
                break
                
        # Scan from back to front to find the last different position
        last_diff = None
        for i in range(1, min_len + 1):
            if clean_tokens_list[-i] != corrupted_tokens_list[-i]:
                last_diff = len(clean_tokens_list) - i
                break
        
        # If different parts are found, set the indices of distinct parts
        if first_diff is not None:
            # For clean_tokens, distinct part starts from first_diff
            if last_diff is not None and last_diff >= first_diff:
                clean_subject_indices = list(range(first_diff, last_diff + 1))
            else:
                clean_subject_indices = list(range(first_diff, len(clean_tokens_list)))
        
        # For corrupted_tokens, distinct part starts from first_diff
        if first_diff is not None:
            if last_diff is not None and last_diff >= first_diff:
                corrupted_subject_indices = list(range(first_diff, last_diff + 1))
            else:
                corrupted_subject_indices = list(range(first_diff, len(corrupted_tokens_list)))
        
        print(f"Synthetic data distinct part detection results (after filtering):")
        print(f"First different position: {first_diff}")
        print(f"Last different position: {last_diff}")
        print(f"Clean subject positions: {clean_subject_indices}")
        if clean_subject_indices:
            print(f"Clean subject tokens: {[clean_tokens_str[i] for i in clean_subject_indices]}")
        print(f"Corrupted subject positions: {corrupted_subject_indices}")
        if corrupted_subject_indices:
            print(f"Corrupted subject tokens: {[corrupted_tokens_str[i] for i in corrupted_subject_indices]}")
            
    else:
        # For regular tasks, get distinct parts from task_config
        print("\nDetecting distinct parts using regular task configuration...")
        if 'clean_subject' in task_config and 'corrupted_subject' in task_config:
            clean_subject = task_config['clean_subject']
            corrupted_subject = task_config['corrupted_subject']
            
            print(f"Clean subject in task config: '{clean_subject}'")
            print(f"Corrupted subject in task config: '{corrupted_subject}'")
            
            # More robust search method: search for subject in token strings
            # 1. First convert subject to token sequence
            clean_subject_tokens_str = model.to_str_tokens(clean_subject)
            corrupted_subject_tokens_str = model.to_str_tokens(corrupted_subject)
            
            print(f"Clean subject tokens: {clean_subject_tokens_str}")
            print(f"Corrupted subject tokens: {corrupted_subject_tokens_str}")
            
            # 2. Find the position of clean_subject_tokens_str in clean_tokens_str
            # Since we've already removed padding, we can directly search
            for i in range(len(clean_tokens_str) - len(clean_subject_tokens_str) + 1):
                # Direct comparison of token strings
                match = True
                for j in range(len(clean_subject_tokens_str)):
                    if clean_tokens_str[i+j] != clean_subject_tokens_str[j]:
                        match = False
                        break
                if match:
                    clean_subject_indices = list(range(i, i + len(clean_subject_tokens_str)))
                    break
            
            # 3. Find the position of corrupted_subject_tokens_str in corrupted_tokens_str
            for i in range(len(corrupted_tokens_str) - len(corrupted_subject_tokens_str) + 1):
                # Direct comparison of token strings
                match = True
                for j in range(len(corrupted_subject_tokens_str)):
                    if corrupted_tokens_str[i+j] != corrupted_subject_tokens_str[j]:
                        match = False
                        break
                if match:
                    corrupted_subject_indices = list(range(i, i + len(corrupted_subject_tokens_str)))
                    break
                    
            # If the above method doesn't find anything, try fuzzy matching
            if not clean_subject_indices:
                print("Trying fuzzy matching for clean subject...")
                for i, token in enumerate(clean_tokens_str):
                    if clean_subject.lower() in token.lower():
                        clean_subject_indices.append(i)
                        print(f"Found token containing clean subject at position {i}: '{token}'")
            
            if not corrupted_subject_indices:
                print("Trying fuzzy matching for corrupted subject...")
                for i, token in enumerate(corrupted_tokens_str):
                    if corrupted_subject.lower() in token.lower():
                        corrupted_subject_indices.append(i)
                        print(f"Found token containing corrupted subject at position {i}: '{token}'")
            
            # Last method: compare the differences between the two sequences
            if not clean_subject_indices or not corrupted_subject_indices:
                print("Using sequence comparison method to find distinct parts...")
                # Find different parts in tokens
                min_len = min(len(clean_tokens_str), len(corrupted_tokens_str))
                diff_positions = []
                
                for i in range(min_len):
                    if clean_tokens_str[i] != corrupted_tokens_str[i]:
                        diff_positions.append(i)
                        
                if diff_positions:
                    if not clean_subject_indices:
                        clean_subject_indices = diff_positions
                        print(f"Clean subject positions found by comparison: {clean_subject_indices}")
                    
                    if not corrupted_subject_indices:
                        corrupted_subject_indices = diff_positions
                        print(f"Corrupted subject positions found by comparison: {corrupted_subject_indices}")
            
            print(f"Final clean subject positions: {clean_subject_indices}")
            if clean_subject_indices:
                print(f"Clean subject tokens: {[clean_tokens_str[i] for i in clean_subject_indices]}")
            print(f"Final corrupted subject positions: {corrupted_subject_indices}")
            if corrupted_subject_indices:
                print(f"Corrupted subject tokens: {[corrupted_tokens_str[i] for i in corrupted_subject_indices]}")
        else:
            print("Warning: Task config does not have clean_subject and corrupted_subject fields.")
            # Use sequence comparison method to find distinct parts
            print("Using sequence comparison method to find distinct parts...")
            # Find different parts in tokens
            min_len = min(len(clean_tokens_str), len(corrupted_tokens_str))
            diff_positions = []
            
            for i in range(min_len):
                if clean_tokens_str[i] != corrupted_tokens_str[i]:
                    diff_positions.append(i)
                    
            if diff_positions:
                clean_subject_indices = diff_positions
                corrupted_subject_indices = diff_positions
                print(f"Distinct positions found by comparison: {diff_positions}")
                print(f"Corresponding clean tokens: {[clean_tokens_str[i] for i in clean_subject_indices]}")
                print(f"Corresponding corrupted tokens: {[corrupted_tokens_str[i] for i in corrupted_subject_indices]}")
    
    # If no distinct parts are found, we return empty results
    if not clean_subject_indices and not corrupted_subject_indices:
        print("Warning: No distinct parts found, cannot perform attention analysis.")
        return {}
    
    # Analyze attention patterns for each attention head
    attention_metrics = {
        'attention_heads': len(attention_nodes),
        'clean_subject_attention_scores': [],
        'corrupted_subject_attention_scores': [],
        'attention_heads_details': [],
        'epoch': current_epoch  # Add epoch information to metrics dictionary
    }
    
    for node_name in attention_nodes:
        try:
            # Parse layer number and head number
            layer_str, head_str = node_name.split('.')
            layer = int(layer_str[1:])  # Remove 'a' prefix
            head = int(head_str[1:])    # Remove 'h' prefix
            
            # Get attention pattern (from corresponding cache)
            attn_key = utils.get_act_name("pattern", layer)
            if attn_key not in clean_cache or attn_key not in corrupted_cache:
                print(f"Warning: Attention pattern key '{attn_key}' not found in cache for node {node_name}.")
                continue
                
            clean_attention_pattern = clean_cache[attn_key][0, head].cpu().numpy()
            corrupted_attention_pattern = corrupted_cache[attn_key][0, head].cpu().numpy()

            # Adjust attention patterns if BOS was present
            if clean_bos_present and clean_attention_pattern.ndim == 2 and clean_attention_pattern.shape[0] > 1 and clean_attention_pattern.shape[1] > 1:
                clean_attention_pattern = clean_attention_pattern[1:, 1:]
            
            if corrupted_bos_present and corrupted_attention_pattern.ndim == 2 and corrupted_attention_pattern.shape[0] > 1 and corrupted_attention_pattern.shape[1] > 1:
                corrupted_attention_pattern = corrupted_attention_pattern[1:, 1:]

            # Calculate attention metrics
            head_metrics = {
                'node_name': node_name,
                'layer': layer,
                'head': head,
                'clean_subject_attention': {},
                'corrupted_subject_attention': {}
            }
            
            # For each query position, calculate the proportion of attention to distinct parts as keys
            seq_len_clean = clean_attention_pattern.shape[0]
            seq_len_corrupted = corrupted_attention_pattern.shape[0] # Use shape of (potentially stripped) pattern
            
            # Calculate for clean subject
            if clean_subject_indices:
                clean_subject_attention_by_query = {}
                
                # Directly use clean subject positions and all queries after
                min_subject_idx = min(clean_subject_indices) if clean_subject_indices else 0
                valid_query_positions = list(range(min_subject_idx, seq_len_clean))
                
                # For each valid query position, calculate attention to subject tokens
                for q_idx in valid_query_positions:
                    if q_idx < clean_attention_pattern.shape[0]:
                        # Calculate total attention from this query position to clean subject
                        valid_key_indices = [idx for idx in clean_subject_indices 
                                            if idx < clean_attention_pattern.shape[1]]
                        
                        if valid_key_indices:
                            subject_attention_sum = sum(clean_attention_pattern[q_idx, key_idx] 
                                                    for key_idx in valid_key_indices)
                            
                            # Calculate total attention for this query position
                            total_attention = sum(clean_attention_pattern[q_idx])
                                
                            # Calculate proportion
                            if total_attention > 0:
                                attention_ratio = subject_attention_sum / total_attention
                                clean_subject_attention_by_query[q_idx] = attention_ratio
                
                # Calculate average attention across all valid query positions
                if clean_subject_attention_by_query:
                    avg_attention = sum(clean_subject_attention_by_query.values()) / len(clean_subject_attention_by_query)
                    clean_subject_attention_by_query['avg_all_queries'] = avg_attention
                        
                    # Store in head metrics
                    head_metrics['clean_subject_attention'] = {
                        'by_query': clean_subject_attention_by_query,
                        'average': avg_attention,
                    }
                        
                    # Add to overall metrics
                    attention_metrics['clean_subject_attention_scores'].append(avg_attention)
            
            # Do similar calculations for corrupted subject
            if corrupted_subject_indices:
                corrupted_subject_attention_by_query = {}
                
                # Directly use corrupted subject positions and all queries after
                min_subject_idx = min(corrupted_subject_indices) if corrupted_subject_indices else 0
                valid_query_positions = list(range(min_subject_idx, seq_len_corrupted))
                
                # For each valid query position, calculate attention to subject tokens
                for q_idx in valid_query_positions:
                    if q_idx < corrupted_attention_pattern.shape[0]:
                        # Calculate total attention from this query position to corrupted subject
                        valid_key_indices = [idx for idx in corrupted_subject_indices 
                                            if idx < corrupted_attention_pattern.shape[1]]
                        
                        if valid_key_indices:
                            subject_attention_sum = sum(corrupted_attention_pattern[q_idx, key_idx] 
                                                      for key_idx in valid_key_indices)
                            
                            # Calculate total attention for this query position
                            total_attention = sum(corrupted_attention_pattern[q_idx])
                                
                            # Calculate proportion
                            if total_attention > 0:
                                attention_ratio = subject_attention_sum / total_attention
                                corrupted_subject_attention_by_query[q_idx] = attention_ratio
                
                # Calculate average attention across all valid query positions
                if corrupted_subject_attention_by_query:
                    avg_attention = sum(corrupted_subject_attention_by_query.values()) / len(corrupted_subject_attention_by_query)
                    corrupted_subject_attention_by_query['avg_all_queries'] = avg_attention
                        
                    # Store in head metrics
                    head_metrics['corrupted_subject_attention'] = {
                        'by_query': corrupted_subject_attention_by_query,
                        'average': avg_attention,
                    }
                        
                    # Add to overall metrics
                    attention_metrics['corrupted_subject_attention_scores'].append(avg_attention)
            
            attention_metrics['attention_heads_details'].append(head_metrics)
            
            # Save attention pattern heatmaps only if save_plots is True
            if save_plots:
                heatmap_dir = os.path.join(output_dir, "attention_heatmaps")
                os.makedirs(heatmap_dir, exist_ok=True)
                
                # Prepare token labels - use actual token text instead of ID format
                # These token lists are already processed (removed padding)
                # For synthetic data, we still use IDs
                if not is_synthetic:
                    clean_token_labels = clean_tokens_str[:clean_attention_pattern.shape[1]]
                    corrupted_token_labels = corrupted_tokens_str[:corrupted_attention_pattern.shape[1]]
                else:
                    # For synthetic data, use ID format
                    clean_token_labels = [f"ID:{clean_tokens_list[i]}" for i in range(len(clean_tokens_list)) if i < clean_attention_pattern.shape[1]]
                    corrupted_token_labels = [f"ID:{corrupted_tokens_list[i]}" for i in range(len(corrupted_tokens_list)) if i < corrupted_attention_pattern.shape[1]]
                
                # Add epoch information to filename
                epoch_suffix = f"_epoch{current_epoch}" if current_epoch is not None else ""
                
                # Clean mode heatmap
                clean_heatmap_filename = os.path.join(heatmap_dir, f"{task_name}_{node_name}_clean_attention{epoch_suffix}.png")
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(clean_attention_pattern, cmap='viridis', 
                            xticklabels=clean_token_labels if len(clean_token_labels) == clean_attention_pattern.shape[1] else False,
                            yticklabels=clean_token_labels if len(clean_token_labels) == clean_attention_pattern.shape[0] else False)
                
                # Add epoch information to title
                epoch_title = f" (Epoch {current_epoch})" if current_epoch is not None else ""
                plt.title(f"Clean Attention Pattern - {node_name} (Layer {layer}, Head {head}){epoch_title}")
                plt.xlabel("Keys")
                plt.ylabel("Queries")
                
                # Highlight distinct parts (subject tokens)
                if clean_subject_indices:
                    for idx in clean_subject_indices:
                        if idx < clean_attention_pattern.shape[1]:  # Ensure within valid range
                            plt.axvline(x=idx + 0.5, color='red', linestyle='--', alpha=0.5, label='Clean Subject' if idx == clean_subject_indices[0] else "")
                
                # Highlight valid query positions we used for attention calculation
                if 'clean_subject_attention' in head_metrics and 'by_query' in head_metrics['clean_subject_attention']:
                    query_positions = [qp for qp in head_metrics['clean_subject_attention']['by_query'].keys() 
                                    if qp != 'avg_all_queries' and qp < clean_attention_pattern.shape[0]]
                    for qp in query_positions:
                        plt.axhline(y=qp + 0.5, color='green', linestyle='-', alpha=0.5, 
                                  label='Query Position' if qp == query_positions[0] else "")
                
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig(clean_heatmap_filename)
                plt.close()
                
                # Corrupted mode heatmap
                corrupted_heatmap_filename = os.path.join(heatmap_dir, f"{task_name}_{node_name}_corrupted_attention{epoch_suffix}.png")
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(corrupted_attention_pattern, cmap='viridis', 
                            xticklabels=corrupted_token_labels if len(corrupted_token_labels) == corrupted_attention_pattern.shape[1] else False,
                            yticklabels=corrupted_token_labels if len(corrupted_token_labels) == corrupted_attention_pattern.shape[0] else False)
                
                # Add epoch information to title
                plt.title(f"Corrupted Attention Pattern - {node_name} (Layer {layer}, Head {head}){epoch_title}")
                plt.xlabel("Keys")
                plt.ylabel("Queries")
                
                # Highlight distinct parts (subject tokens)
                if corrupted_subject_indices:
                    for idx in corrupted_subject_indices:
                        if idx < corrupted_attention_pattern.shape[1]:  # Ensure within valid range
                            plt.axvline(x=idx + 0.5, color='green', linestyle=':', alpha=0.5, label='Corrupted Subject' if idx == corrupted_subject_indices[0] else "")
                
                # Highlight valid query positions we used for attention calculation
                if 'corrupted_subject_attention' in head_metrics and 'by_query' in head_metrics['corrupted_subject_attention']:
                    query_positions = [qp for qp in head_metrics['corrupted_subject_attention']['by_query'].keys() 
                                    if qp != 'avg_all_queries' and qp < corrupted_attention_pattern.shape[0]]
                    for qp in query_positions:
                        plt.axhline(y=qp + 0.5, color='green', linestyle='-', alpha=0.5, 
                                  label='Query Position' if qp == query_positions[0] else "")
                
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig(corrupted_heatmap_filename)
                plt.close()
            
        except Exception as e:
            print(f"Error analyzing attention pattern for node {node_name}: {e}")
            traceback.print_exc()
    
    # Calculate overall metrics
    if attention_metrics['clean_subject_attention_scores']:
        attention_metrics['avg_clean_subject_attention'] = sum(attention_metrics['clean_subject_attention_scores']) / len(attention_metrics['clean_subject_attention_scores'])
    else:
        attention_metrics['avg_clean_subject_attention'] = 0.0
        
    if attention_metrics['corrupted_subject_attention_scores']:
        attention_metrics['avg_corrupted_subject_attention'] = sum(attention_metrics['corrupted_subject_attention_scores']) / len(attention_metrics['corrupted_subject_attention_scores'])
    else:
        attention_metrics['avg_corrupted_subject_attention'] = 0.0
    
    # Calculate difference metric (measuring degree of attention preference for the two distinct parts)
    if attention_metrics['clean_subject_attention_scores'] and attention_metrics['corrupted_subject_attention_scores']:
        attention_metrics['subject_attention_contrast'] = attention_metrics['avg_clean_subject_attention'] - attention_metrics['avg_corrupted_subject_attention']
    else:
        attention_metrics['subject_attention_contrast'] = 0.0
    
    # Output summary metrics
    print("\n===== Attention Pattern Analysis Summary (New Method) =====")
    if current_epoch is not None:
        print(f"Current Epoch: {current_epoch}")
    print(f"Total attention heads analyzed: {len(attention_nodes)}")
    print(f"Average attention to clean subject: {attention_metrics['avg_clean_subject_attention']:.4f} ({len(attention_metrics['clean_subject_attention_scores'])} heads)")
    print(f"Average attention to corrupted subject: {attention_metrics['avg_corrupted_subject_attention']:.4f} ({len(attention_metrics['corrupted_subject_attention_scores'])} heads)")
    print(f"Subject attention contrast (clean - corrupted): {attention_metrics['subject_attention_contrast']:.4f}")
    print("Attention calculation now uses: average across all valid query positions after subject starts (excluding padding)")
    print("=============================================\n")
    
    # Output detailed metrics for each attention head
    print("\n----- Detailed Metrics for Each Attention Head -----")
    for head_detail in attention_metrics['attention_heads_details']:
        node_name = head_detail['node_name']
        clean_avg = head_detail['clean_subject_attention'].get('average', 0.0) if 'clean_subject_attention' in head_detail else 0.0
        corrupted_avg = head_detail['corrupted_subject_attention'].get('average', 0.0) if 'corrupted_subject_attention' in head_detail else 0.0
        
        # Calculate contrast
        avg_contrast = clean_avg - corrupted_avg
        
        print(f"Head {node_name}: Clean Avg={clean_avg:.4f}, Corrupted Avg={corrupted_avg:.4f}, Contrast={avg_contrast:.4f}")
    print("---------------------------------------------\n")

    # --- High attention head analysis started ---
    high_attention_threshold = 0.20
    high_attention_heads_details = []
    high_attention_clean_scores = []
    high_attention_corrupted_scores = []

    for head_detail in attention_metrics['attention_heads_details']:
        clean_avg = head_detail['clean_subject_attention'].get('average', 0.0) if 'clean_subject_attention' in head_detail else 0.0
        corrupted_avg = head_detail['corrupted_subject_attention'].get('average', 0.0) if 'corrupted_subject_attention' in head_detail else 0.0
        if clean_avg > high_attention_threshold or corrupted_avg > high_attention_threshold:
            high_attention_heads_details.append(head_detail)
            if 'clean_subject_attention' in head_detail and 'average' in head_detail['clean_subject_attention']:
                 high_attention_clean_scores.append(head_detail['clean_subject_attention']['average'])
            if 'corrupted_subject_attention' in head_detail and 'average' in head_detail['corrupted_subject_attention']:
                 high_attention_corrupted_scores.append(head_detail['corrupted_subject_attention']['average'])
    
    attention_metrics['high_attention_heads_details'] = high_attention_heads_details
    attention_metrics['num_high_attention_heads'] = len(high_attention_heads_details)

    if high_attention_clean_scores:
        attention_metrics['avg_high_attention_clean_subject_attention'] = sum(high_attention_clean_scores) / len(high_attention_clean_scores)
    else:
        attention_metrics['avg_high_attention_clean_subject_attention'] = 0.0
        
    if high_attention_corrupted_scores:
        attention_metrics['avg_high_attention_corrupted_subject_attention'] = sum(high_attention_corrupted_scores) / len(high_attention_corrupted_scores)
    else:
        attention_metrics['avg_high_attention_corrupted_subject_attention'] = 0.0
        
    attention_metrics['high_attention_subject_attention_contrast'] = attention_metrics['avg_high_attention_clean_subject_attention'] - attention_metrics['avg_high_attention_corrupted_subject_attention']

    print(f"\n===== High Attention Heads (> {high_attention_threshold*100:.0f}%) Analysis Summary =====")
    if current_epoch is not None:
        print(f"Current Epoch: {current_epoch}")
    print(f"Number of high attention heads: {attention_metrics['num_high_attention_heads']} (out of {len(attention_nodes)} total)")
    print(f"Avg attention to clean subject (High Attn Heads): {attention_metrics['avg_high_attention_clean_subject_attention']:.4f} ({len(high_attention_clean_scores)} heads)")
    print(f"Avg attention to corrupted subject (High Attn Heads): {attention_metrics['avg_high_attention_corrupted_subject_attention']:.4f} ({len(high_attention_corrupted_scores)} heads)")
    print(f"Subject attention contrast (High Attn Heads): {attention_metrics['high_attention_subject_attention_contrast']:.4f}")
    print("========================================================\n")

    # Save high-attention metrics to a separate JSON file
    if high_attention_heads_details:
        high_attention_metrics_to_save = {
            'epoch': current_epoch,
            'task_name': task_name,
            'num_total_attention_heads': len(attention_nodes),
            'high_attention_threshold': high_attention_threshold,
            'num_high_attention_heads': attention_metrics['num_high_attention_heads'],
            'avg_high_attention_clean_subject_attention': attention_metrics['avg_high_attention_clean_subject_attention'],
            'avg_high_attention_corrupted_subject_attention': attention_metrics['avg_high_attention_corrupted_subject_attention'],
            'high_attention_subject_attention_contrast': attention_metrics['high_attention_subject_attention_contrast'],
            'high_attention_heads_details': high_attention_heads_details
        }
        high_attention_json_filename = os.path.join(output_dir, f"{task_name}_high_attention_metrics_epoch{current_epoch}.json")
        try:
            with open(high_attention_json_filename, 'w') as f_high:
                json.dump(high_attention_metrics_to_save, f_high, indent=2)
            print(f"High attention metrics saved to: {high_attention_json_filename}")
            if not save_plots:
                print("Note: Attention images saved disabled, only saved metadata JSON file")
        except Exception as e_json_high:
            print(f"Error saving high attention metrics JSON: {e_json_high}")
    # --- High attention head analysis ended ---
    
    # Draw bar charts for overall metrics
    if save_plots:
        metrics_dir = os.path.join(output_dir, "attention_metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Add epoch information to filename
        metrics_suffix = f"_epoch{current_epoch}" if current_epoch is not None else ""
        
        # Draw average attention allocation bar chart (using new calculation method)
        avg_metrics_filename = os.path.join(metrics_dir, f"{task_name}_average_attention{metrics_suffix}.png")
        
        plt.figure(figsize=(12, 8))
        heads = attention_metrics['attention_heads_details']
        nodes = [head['node_name'] for head in heads]
        avg_clean_scores = [head['clean_subject_attention'].get('average', 0.0) if 'clean_subject_attention' in head else 0.0 for head in heads]
        avg_corrupted_scores = [head['corrupted_subject_attention'].get('average', 0.0) if 'corrupted_subject_attention' in head else 0.0 for head in heads]
        
        x = np.arange(len(nodes))
        width = 0.35
        
        plt.bar(x - width/2, avg_clean_scores, width, label='Clean Subject')
        plt.bar(x + width/2, avg_corrupted_scores, width, label='Corrupted Subject')
        
        plt.xlabel('Attention Heads')
        plt.ylabel('Average Attention Ratio')
        
        # Add epoch information to title
        epoch_plot_title = f" (Epoch {current_epoch})" if current_epoch is not None else ""
        plt.title(f'Attention to Subject Areas - {task_name}{epoch_plot_title}\n(Using New Method: Average Across All Valid Query Positions)')
        plt.xticks(x, nodes, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add average value lines
        plt.axhline(y=attention_metrics['avg_clean_subject_attention'], color='blue', linestyle='-', alpha=0.5, label='Avg Clean')
        plt.axhline(y=attention_metrics['avg_corrupted_subject_attention'], color='orange', linestyle='-', alpha=0.5, label='Avg Corrupted')
        
        plt.tight_layout()
        plt.savefig(avg_metrics_filename)
        plt.close()
    
    return attention_metrics