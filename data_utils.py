import torch
import re
import json
def parse_synthetic_line(line_str):
    """Parse synthetic data line using regex to extract token lists."""
    data = {}
    pattern = re.compile(r'(\w+):\[([\d\s,]*?)\]')
    matches = pattern.findall(line_str)
    
    for key, value_str in matches:
        try:
            tokens = [int(x.strip()) for x in value_str.split(',') if x.strip()]
            data[key] = tokens
        except ValueError:
            data[key] = []
    return data

def load_and_parse_synthetic_data(file_path, target_prompt_tokens):
    """Load synthetic data file and find entries matching target prompt tokens."""
    matched_lines_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_data = parse_synthetic_line(line.strip())
                if line_data.get("Xshare") == target_prompt_tokens:
                    matched_lines_data.append(line_data)
    except FileNotFoundError:
        print(f"Error: Synthetic data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading or parsing synthetic data file {file_path}: {e}")
        return None
    
    return matched_lines_data

def align_subject_tokens(model, clean_subject, corrupted_subject):
    """Align token sequences to ensure equal length by padding with <unk> tokens."""
    clean_tokens = model.to_tokens(clean_subject, prepend_bos=True)[0]
    corrupted_tokens = model.to_tokens(corrupted_subject, prepend_bos=True)[0]
    
    clean_len = len(clean_tokens)
    corrupted_len = len(corrupted_tokens)
    
    if clean_len == corrupted_len:
        return clean_tokens, corrupted_tokens, None
    
    # Find appropriate padding token ID
    unk_token_id = None
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        if hasattr(model.tokenizer, 'unk_token_id') and model.tokenizer.unk_token_id is not None:
            unk_token_id = model.tokenizer.unk_token_id
    
    if unk_token_id is None:
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            if hasattr(model.tokenizer, 'pad_token_id') and model.tokenizer.pad_token_id is not None:
                unk_token_id = model.tokenizer.pad_token_id
    
    if unk_token_id is None:
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            if hasattr(model.tokenizer, 'eos_token_id') and model.tokenizer.eos_token_id is not None:
                unk_token_id = model.tokenizer.eos_token_id
    
    if unk_token_id is None:
        special_tokens = [0, 1, 2, 3]
        for token_id in special_tokens:
            if token_id < model.cfg.d_vocab:
                unk_token_id = token_id
                break
    
    if unk_token_id is None:
        unk_token_id = 0
    
    # Align sequences while preserving BOS token at the beginning
    if clean_len < corrupted_len:
        diff = corrupted_len - clean_len
        
        if clean_tokens.shape[0] > 0:
            bos_token = clean_tokens[0:1]
            rest_tokens = clean_tokens[1:] if clean_len > 1 else torch.tensor([], dtype=clean_tokens.dtype, device=clean_tokens.device)
            padding = torch.full((diff,), unk_token_id, dtype=clean_tokens.dtype, device=clean_tokens.device)
            aligned_clean_tokens = torch.cat([bos_token, padding, rest_tokens])
        else:
            padding = torch.full((diff,), unk_token_id, dtype=clean_tokens.dtype, device=clean_tokens.device)
            aligned_clean_tokens = torch.cat([padding, clean_tokens])
            
        aligned_corrupted_tokens = corrupted_tokens
    else:
        diff = clean_len - corrupted_len
        
        if corrupted_tokens.shape[0] > 0:
            bos_token = corrupted_tokens[0:1]
            rest_tokens = corrupted_tokens[1:] if corrupted_len > 1 else torch.tensor([], dtype=corrupted_tokens.dtype, device=corrupted_tokens.device)
            padding = torch.full((diff,), unk_token_id, dtype=corrupted_tokens.dtype, device=corrupted_tokens.device)
            aligned_corrupted_tokens = torch.cat([bos_token, padding, rest_tokens])
        else:
            padding = torch.full((diff,), unk_token_id, dtype=corrupted_tokens.dtype, device=corrupted_tokens.device)
            aligned_corrupted_tokens = torch.cat([padding, corrupted_tokens])
            
        aligned_clean_tokens = clean_tokens
    
    assert len(aligned_clean_tokens) == len(aligned_corrupted_tokens), "Token lengths not equal after alignment"
    
    if aligned_clean_tokens.shape[0] > 0 and aligned_corrupted_tokens.shape[0] > 0:
        if aligned_clean_tokens[0].item() != aligned_corrupted_tokens[0].item():
            pass
    
    return aligned_clean_tokens, aligned_corrupted_tokens, unk_token_id

def prepare_task_data(model, task_config, mode=2, use_synthetic_dataset=False, synthetic_data_file_path=None):
    """Prepare task data with token alignment. Supports both synthetic and regular datasets."""
    if use_synthetic_dataset:
        if not synthetic_data_file_path:
            raise ValueError("Synthetic data file path must be provided when use_synthetic_dataset is True.")
        if "Prompt Tokens" not in task_config:
            raise ValueError("Synthetic task_config must contain 'Prompt Tokens'.")

        prompt_tokens_to_find = task_config["Prompt Tokens"]
        matched_data_entries = load_and_parse_synthetic_data(synthetic_data_file_path, prompt_tokens_to_find)

        if not matched_data_entries:
            raise ValueError(f"No synthetic data found for Prompt Tokens {prompt_tokens_to_find} in {synthetic_data_file_path}")

        xa_entry = None
        xb_entry = None

        for entry in matched_data_entries:
            if "Xa" in entry and "Ya" in entry:
                xa_entry = entry
            if "Xb" in entry and "Yb" in entry:
                xb_entry = entry
        
        if not xa_entry:
            raise ValueError(f"Corrupted data (Xa/Ya) not found for Prompt Tokens {prompt_tokens_to_find} in matched entries.")
        if not xb_entry:
            raise ValueError(f"Clean data (Xb/Yb) not found for Prompt Tokens {prompt_tokens_to_find} in matched entries.")

        x_share_tokens = xb_entry["Xshare"]
        xa_tokens = xa_entry["Xa"]
        ya_token_id = xa_entry["Ya"][0]
        xb_tokens = xb_entry["Xb"]
        yb_token_id = xb_entry["Yb"][0]

        clean_tokens_list = x_share_tokens + xb_tokens
        corrupted_tokens_list = x_share_tokens + xa_tokens
        # Add BOS token if available for consistency
        bos_token_id = None
        if hasattr(model, 'tokenizer') and model.tokenizer is not None and hasattr(model.tokenizer, 'bos_token_id'):
            bos_token_id = model.tokenizer.bos_token_id
        
        if bos_token_id is not None:
            final_clean_tokens_list = [bos_token_id] + clean_tokens_list
            final_corrupted_tokens_list = [bos_token_id] + corrupted_tokens_list
        else:
            final_clean_tokens_list = clean_tokens_list
            final_corrupted_tokens_list = corrupted_tokens_list

        device = model.cfg.device
        clean_tokens = torch.tensor(final_clean_tokens_list, dtype=torch.long, device=device).unsqueeze(0)
        corrupted_tokens = torch.tensor(final_corrupted_tokens_list, dtype=torch.long, device=device).unsqueeze(0)
        label_tensor = torch.tensor([[yb_token_id, ya_token_id]], dtype=torch.long, device=device)

        try:
            clean_prompt_text = model.to_string(clean_tokens[0])
            corrupted_prompt_text = model.to_string(corrupted_tokens[0])
        except Exception:
            clean_prompt_text = f"[Synthetic Clean Tokens: {final_clean_tokens_list}]"
            corrupted_prompt_text = f"[Synthetic Corrupted Tokens: {final_corrupted_tokens_list}]"
        
        return clean_tokens, corrupted_tokens, label_tensor, clean_prompt_text, corrupted_prompt_text

    else:
        # Handle regular task configs
        if mode == 0:
            clean_subject = task_config["clean_subject"]
            prompt_template = task_config["prompt_template"]
            clean_prompt = prompt_template.format(clean_subject)
            corrupted_prompt = task_config.get("irrelevant_prompt", "Default irrelevant prompt if missing")
            correct_answer = task_config["correct_answer"]
            incorrect_answer = task_config.get("irrelevant_answer", "Default irrelevant answer if missing")
        elif mode == 1:
            clean_subject = task_config["corrupted_subject"]
            prompt_template = task_config["prompt_template"]
            clean_prompt = prompt_template.format(clean_subject)
            corrupted_prompt = task_config.get("irrelevant_prompt", "Default irrelevant prompt if missing")
            correct_answer = task_config["incorrect_answer"]
            incorrect_answer = task_config.get("irrelevant_answer", "Default irrelevant answer if missing")
        else:
            clean_subject = task_config["clean_subject"]
            corrupted_subject = task_config["corrupted_subject"]
            prompt_template = task_config["prompt_template"]
            clean_prompt = prompt_template.format(clean_subject)
            corrupted_prompt = prompt_template.format(corrupted_subject)
            correct_answer = task_config["correct_answer"]
            incorrect_answer = task_config["incorrect_answer"]
        
        aligned_clean_tokens_no_batch, aligned_corrupted_tokens_no_batch, _ = align_subject_tokens(model, clean_prompt, corrupted_prompt)
        
        correct_answer_tokens = model.to_tokens(correct_answer, prepend_bos=False)
        if correct_answer_tokens.shape[1] == 0:
            raise ValueError(f"Error: Correct answer '{correct_answer}' tokenization failed!")
        correct_idx = correct_answer_tokens[0, 0].item()

        incorrect_answer_tokens = model.to_tokens(incorrect_answer, prepend_bos=False)
        if incorrect_answer_tokens.shape[1] == 0:
            raise ValueError(f"Error: Incorrect answer '{incorrect_answer}' tokenization failed!")
        incorrect_idx = incorrect_answer_tokens[0, 0].item()

        label_tensor = torch.tensor([[correct_idx, incorrect_idx]], device=model.cfg.device)

        clean_tokens_batch = aligned_clean_tokens_no_batch.unsqueeze(0)
        corrupted_tokens_batch = aligned_corrupted_tokens_no_batch.unsqueeze(0)
        
        return clean_tokens_batch, corrupted_tokens_batch, label_tensor, clean_prompt, corrupted_prompt 