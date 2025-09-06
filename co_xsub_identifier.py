import torch
import math
import numpy as np
from tqdm import tqdm

def get_top_k_predictions(prompt_text, model, tokenizer, top_k=10):
    tokens = safe_to_tokens(model, prompt_text)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    
    model_device = next(model.parameters()).device
    tokens = tokens.to(model_device)
    with torch.no_grad():
        logits = model(tokens)
    next_token_logits = logits[:, -1, :]
    log_probs = torch.log_softmax(next_token_logits, dim=-1)
    top_k_log_probs, top_k_indices = torch.topk(log_probs, top_k, dim=-1)
    top_k_tokens = []
    for idx in top_k_indices[0]:
        token_str = model.to_string([idx.item()])
        if isinstance(token_str, list):
            token_str = token_str[0]
        top_k_tokens.append(token_str)
    predictions = []
    for token_str, log_prob_val in zip(top_k_tokens, top_k_log_probs[0]):
        predictions.append((token_str, log_prob_val.item()))
    return predictions

def compute_logprob_variance_across_masks(prompt_tokens, model, tokenizer, original_predictions_list, top_k=10):
    from collections import defaultdict
    token_logprobs = defaultdict(list)
    for i in range(len(prompt_tokens)):
        temp_prompt_tokens = prompt_tokens[:i] + prompt_tokens[i+1:]
        masked_prompt = "".join(temp_prompt_tokens) if hasattr(model, 'to_string') else tokenizer.convert_tokens_to_string(temp_prompt_tokens)
        if not masked_prompt.strip():
            continue
        masked_predictions_list = get_top_k_predictions(masked_prompt, model, tokenizer, top_k=top_k)
        masked_predictions_map = {token: log_p for token, log_p in masked_predictions_list}
        for token, _ in original_predictions_list:
            log_p = masked_predictions_map.get(token, -10.0)
            token_logprobs[token].append(log_p)
    token_var = {}
    for token, logprobs in token_logprobs.items():
        if len(logprobs) > 1:
            token_var[token] = float(np.var(logprobs))
        else:
            token_var[token] = 0.0
    return token_var

def _is_eos_token(token_str, model):
    eos_token = None
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'eos_token'):
        eos_token = model.tokenizer.eos_token
    if eos_token is None:
        eos_token = '<|endoftext|>'
    return token_str == eos_token

def clean_prompt_tokens(prompt_tokens, model):
    eos_token = None
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'eos_token'):
        eos_token = model.tokenizer.eos_token
    if eos_token is None:
        eos_token = '<|endoftext|>'
    new_tokens = []
    found_eos = False
    for i, t in enumerate(prompt_tokens):
        if t == eos_token:
            if not found_eos:
                new_tokens.append(t)
                found_eos = True
        else:
            new_tokens.append(t)
    return new_tokens

def safe_to_str_tokens(model, prompt):
    tokens = model.to_str_tokens(prompt)
    tokens = clean_prompt_tokens(tokens, model)
    return tokens

def safe_to_tokens(model, prompt):
    return model.to_tokens(prompt, prepend_bos=False)

def identify_xsub_by_phantomcircuit(prompt, model, tokenizer, top_k=10, verbose=True):
    prompt_tokens = safe_to_str_tokens(model, prompt)
    original_predictions_list = get_top_k_predictions(prompt, model, tokenizer, top_k=top_k)
    original_predictions_map = {token: log_p for token, log_p in original_predictions_list}
    logprob_var_map = compute_logprob_variance_across_masks(prompt_tokens, model, tokenizer, original_predictions_list, top_k=top_k)
    s_rpmi_scores_for_x_sub = []
    if verbose:
        print("\n--- Identifying X_sub (phantomcircuit/CoDA) ---")
        print(f"Prompt: {prompt}")
        print(f"Tokenized prompt: {prompt_tokens}")
        print(f"Top-{top_k} predictions for prompt:")
        for i, (token, logp) in enumerate(original_predictions_list):
            prob = math.exp(logp)
            print(f"  {i+1:2d}: '{token}'  log_prob: {logp:.4f}  prob: {prob:.4%}")
    for i in tqdm(range(len(prompt_tokens)), desc="Masking for X_sub"):
        masked_token_candidate = prompt_tokens[i]
        if _is_eos_token(masked_token_candidate, model):
            continue
        temp_prompt_tokens = prompt_tokens[:i] + prompt_tokens[i+1:]
        masked_prompt_P_prime_sub = "".join(temp_prompt_tokens) if hasattr(model, 'to_string') else tokenizer.convert_tokens_to_string(temp_prompt_tokens)
        if not masked_prompt_P_prime_sub.strip():
            s_rpmi_scores_for_x_sub.append({'token': masked_token_candidate, 's_rpmi': 0, 'masked_prompt':masked_prompt_P_prime_sub})
            continue
        masked_predictions_list = get_top_k_predictions(masked_prompt_P_prime_sub, model, tokenizer, top_k=top_k)
        masked_predictions_map = {token: log_p for token, log_p in masked_predictions_list}
        weighted_srpmi = 0.0
        for token, log_p in original_predictions_map.items():
            log_p_masked = masked_predictions_map.get(token, -10.0)
            rpmi = log_p - log_p_masked
            srpmi = min(0, rpmi)
            weight = logprob_var_map.get(token, 0.0)
            weighted = srpmi * weight
            weighted_srpmi += weighted
        s_rpmi_scores_for_x_sub.append({'token': masked_token_candidate, 's_rpmi': weighted_srpmi, 'masked_prompt': masked_prompt_P_prime_sub})
    if not s_rpmi_scores_for_x_sub:
        if verbose:
            print("Could not calculate S_R-PMI scores for X_sub identification.")
        return None, -1
    best_x_sub_candidate = min(s_rpmi_scores_for_x_sub, key=lambda x: x['s_rpmi'])
    X_sub = best_x_sub_candidate['token']
    X_sub_idx = -1
    for i, token in enumerate(prompt_tokens):
        if token == X_sub:
            X_sub_idx = i
            break
    if verbose:
        print(f"\nIdentified X_sub: '{X_sub}' (S_R-PMI: {best_x_sub_candidate['s_rpmi']:.4f})")
        print(f"   All candidates and S_R-PMI: {[{'token': x['token'], 's_rpmi': x['s_rpmi']} for x in s_rpmi_scores_for_x_sub]}")
    return X_sub, X_sub_idx

def identify_xsub_by_complex(prompt, model, tokenizer, top_k=10, alpha_vtop=0.01, verbose=True):
    prompt_tokens = safe_to_str_tokens(model, prompt)
    if verbose:
        print("\n--- Identifying X_sub (complex/Co_law method) ---")
        print(f"Prompt: {prompt}")
        print(f"Tokenized prompt: {prompt_tokens}")
    def get_all_log_probs_and_vtop(prompt_text, model, tokenizer, alpha=alpha_vtop):
        tokens = safe_to_tokens(model, prompt_text)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        model_device = next(model.parameters()).device
        tokens = tokens.to(model_device)
        with torch.no_grad():
            logits = model(tokens)
        next_token_logits = logits[:, -1, :]
        all_log_probs = torch.log_softmax(next_token_logits, dim=-1).squeeze()
        max_log_prob_val = torch.max(all_log_probs).item()
        log_alpha = math.log(alpha) if alpha > 0 else -float('inf')
        threshold = log_alpha + max_log_prob_val
        v_top_indices = torch.where(all_log_probs >= threshold)[0].tolist()
        return all_log_probs, set(v_top_indices)
    all_log_probs_X, v_top_X = get_all_log_probs_and_vtop(prompt, model, tokenizer, alpha=alpha_vtop)
    indicator_scores_for_x_sub = []
    if verbose:
        top_k_indices = torch.topk(all_log_probs_X, top_k).indices
        print(f"Top-{top_k} predictions for prompt:")
        for i, idx in enumerate(top_k_indices):
            token_str = model.to_string([idx.item()])
            if isinstance(token_str, list):
                token_str = token_str[0]
            logp = all_log_probs_X[idx].item()
            prob = math.exp(logp)
            print(f"  {i+1:2d}: '{token_str}'  log_prob: {logp:.4f}  prob: {prob:.4%}")
    for i in tqdm(range(len(prompt_tokens)), desc="Masking for X_sub (X_b method)"):
        masked_token_candidate_str = prompt_tokens[i]
        if _is_eos_token(masked_token_candidate_str, model):
            continue
        temp_prompt_tokens = prompt_tokens[:i] + prompt_tokens[i+1:]
        masked_prompt_X_prime = "".join(temp_prompt_tokens)
        if not masked_prompt_X_prime.strip():
            indicator_scores_for_x_sub.append({'token_str': masked_token_candidate_str, 'indicator': -float('inf'), 'masked_prompt': ""})
            continue
        all_log_probs_X_prime, v_top_X_prime = get_all_log_probs_and_vtop(masked_prompt_X_prime, model, tokenizer, alpha=alpha_vtop)
        r_pmi_sum = 0.0
        common_v_top = v_top_X.intersection(v_top_X_prime)
        set_S_for_erm = set()
        if common_v_top:
            for token_idx in common_v_top:
                log_p_y_X = all_log_probs_X[token_idx].item()
                log_p_y_X_prime = all_log_probs_X_prime[token_idx].item()
                if math.isinf(log_p_y_X) or math.isinf(log_p_y_X_prime):
                    rpmi = 0
                else:
                    rpmi = log_p_y_X - log_p_y_X_prime
                r_pmi_sum += min(0, rpmi)
                if rpmi < 0:
                    set_S_for_erm.add(token_idx)
        v_esc = v_top_X.difference(v_top_X_prime)
        erm = 0.0
        min_log_p_S_X_prime = float('inf')
        if set_S_for_erm:
            for token_idx_s in set_S_for_erm:
                log_p_val = all_log_probs_X_prime[token_idx_s].item()
                if not math.isinf(log_p_val):
                    min_log_p_S_X_prime = min(min_log_p_S_X_prime, log_p_val)
        if not math.isinf(min_log_p_S_X_prime):
            if v_esc:
                for token_idx_esc in v_esc:
                    log_p_y_esc_X = all_log_probs_X[token_idx_esc].item()
                    if not math.isinf(log_p_y_esc_X):
                        erm += (log_p_y_esc_X - min_log_p_S_X_prime)
        indicator = r_pmi_sum + erm
        indicator_scores_for_x_sub.append({
            'token': masked_token_candidate_str,
            'indicator': indicator,
            'masked_prompt': masked_prompt_X_prime,
            'r_pmi_sum': r_pmi_sum,
            'erm': erm
        })
    if not indicator_scores_for_x_sub:
        if verbose:
            print("Could not calculate Indicator scores for X_sub identification.")
        return None, -1
    best_x_sub_candidate = max(indicator_scores_for_x_sub, key=lambda x: x['indicator'])
    X_sub = best_x_sub_candidate['token']
    X_sub_idx = -1
    for i, token_str_in_prompt in enumerate(prompt_tokens):
        if token_str_in_prompt == X_sub:
            X_sub_idx = i
            break
    if verbose:
        print(f"\nIdentified X_sub (X_b): '{X_sub}' (Max Indicator: {best_x_sub_candidate['indicator']:.4f})")
        print(f"  (R-PMI_sum: {best_x_sub_candidate['r_pmi_sum']:.4f}, ERM: {best_x_sub_candidate['erm']:.4f})")
        print(f"   All candidates and indicators: {[{'token': x['token'], 'indicator': x['indicator']} for x in indicator_scores_for_x_sub]}")
    return X_sub, X_sub_idx

def identify_xsub_by_co(prompt, model, tokenizer, method='phantomcircuit', top_k=10, alpha_vtop=0.01, verbose=True):
    if method == 'phantomcircuit':
        return identify_xsub_by_phantomcircuit(prompt, model, tokenizer, top_k=top_k, verbose=verbose)
    elif method == 'complex':
        return identify_xsub_by_complex(prompt, model, tokenizer, top_k=top_k, alpha_vtop=alpha_vtop, verbose=verbose)
    else:
        raise ValueError(f"Unknown X_sub positioning method: {method}")

def identify_ysub_ydom(prompt, model, tokenizer, correct_answer=None, incorrect_answer=None, top_k=10, verbose=True):
    prompt_tokens = safe_to_str_tokens(model, prompt)
    if verbose:
        print(f"\n--- Automatic Identification of Ysub/Ydom ---")
        print(f"Prompt: {prompt}")
        print(f"Tokenized prompt: {prompt_tokens}")
    original_predictions_list = get_top_k_predictions(prompt, model, tokenizer, top_k=top_k)
    original_ranks_map = {token: rank for rank, (token, _) in enumerate(original_predictions_list)}
    if verbose:
        print(f"\nTop-{top_k} predictions: {original_predictions_list}")
    xsub_token, xsub_idx = identify_xsub_by_co(prompt, model, tokenizer, top_k=top_k, verbose=False)
    if xsub_token is None or xsub_idx == -1:
        print("X_sub not identified, cannot perform Ysub identification.")
        return None, None, {}
    
    ysub_candidates = [token for token, _ in original_predictions_list]
    ysub_stats = []
    import numpy as np
    
    for ysub_cand in ysub_candidates:
        rank_improvements = []
        orig_rank = original_ranks_map[ysub_cand]
        weight = top_k - orig_rank
        
        for i, token in enumerate(prompt_tokens):
            if i == xsub_idx:
                continue
                
            temp_prompt_tokens = prompt_tokens[:i] + prompt_tokens[i+1:]
            masked_prompt = tokenizer.convert_tokens_to_string(temp_prompt_tokens)
            if not masked_prompt.strip():
                continue
                
            masked_predictions_list = get_top_k_predictions(masked_prompt, model, tokenizer, top_k=top_k)
            masked_ranks_map = {token: rank for rank, (token, _) in enumerate(masked_predictions_list)}
            masked_rank = masked_ranks_map.get(ysub_cand, top_k)
            
            rank_improvement = orig_rank - masked_rank
            weighted_rank_improvement = rank_improvement * weight
            rank_improvements.append(weighted_rank_improvement)
            
        if rank_improvements:
            avg_weighted_rank_improvement = np.mean(rank_improvements)
            ysub_stats.append({
                'token': ysub_cand,
                'avg_weighted_rank_improvement': avg_weighted_rank_improvement,
                'orig_rank': orig_rank,
                'weight': weight,
                'rank_improvements': rank_improvements
            })
            if verbose:
                print(f"     Candidate: '{ysub_cand}', Original rank: {orig_rank}, Weight: {weight}, Rank improvements: {rank_improvements}, Avg weighted rank improvement: {avg_weighted_rank_improvement:.2f}")
        else:
            if verbose:
                print(f"     Candidate: '{ysub_cand}', No valid rank improvements after masking")
    
    if ysub_stats:
        ysub_best = max(ysub_stats, key=lambda x: x['avg_weighted_rank_improvement'])
        ysub_token = ysub_best['token']
        if verbose:
            print(f"\nY_sub (Max Avg Weighted Rank Improvement): '{ysub_token}', Score: {ysub_best['avg_weighted_rank_improvement']:.2f}")
    else:
        ysub_token = None
        if verbose:
            print("Y_sub not identified (no valid rank improvements found)")
    ysub_correct = (correct_answer is not None and ysub_token.strip() == correct_answer.strip())
    if verbose and correct_answer is not None:
        print(f"Y_sub identification {'Correct' if ysub_correct else 'Incorrect'}, Identified as: '{ysub_token}', Correct answer: '{correct_answer}'")
    y_dom_candidates = [token for token, _ in original_predictions_list]
    y_dom_stats = []
    import numpy as np
    for y_cand in y_dom_candidates:
        ranks_in_masked = []
        for i, token in enumerate(prompt_tokens):
            temp_prompt_tokens = prompt_tokens[:i] + prompt_tokens[i+1:]
            masked_prompt = tokenizer.convert_tokens_to_string(temp_prompt_tokens)
            if not masked_prompt.strip():
                continue
            masked_predictions_list = get_top_k_predictions(masked_prompt, model, tokenizer, top_k=top_k)
            masked_ranks_map = {token: rank for rank, (token, _) in enumerate(masked_predictions_list)}
            rank = masked_ranks_map.get(y_cand, top_k)
            ranks_in_masked.append(rank)
        if ranks_in_masked:
            avg_rank = np.mean(ranks_in_masked)
            y_dom_stats.append({'token': y_cand, 'avg_rank': avg_rank})
            if verbose:
                print(f"   Candidate: '{y_cand}', Ranks after masking each token: {ranks_in_masked}, Average rank: {avg_rank:.2f}")
        else:
            if verbose:
                print(f"   Candidate: '{y_cand}', No valid ranks after masking")
    if y_dom_stats:
        y_dom = min(y_dom_stats, key=lambda x: x['avg_rank'])
        ydom_token = y_dom['token']
        if verbose:
            print(f"Ydom: '{ydom_token}', Average rank lowest after masking: {y_dom['avg_rank']:.2f}")
    else:
        ydom_token = None
        if verbose:
            print("Ydom not identified (possibly all masked prompts are empty)")
    ydom_correct = (incorrect_answer is not None and ydom_token is not None and ydom_token.strip() == incorrect_answer.strip())
    if verbose and incorrect_answer is not None:
        print(f"Ydom identification {'Correct' if ydom_correct else 'Incorrect'}, Identified as: '{ydom_token}', Incorrect answer: '{incorrect_answer}'")
    detail = {
        'ysub_token': ysub_token,
        'ysub_correct': ysub_correct,
        'ydom_token': ydom_token,
        'ydom_correct': ydom_correct,
        'ysub_detail': ysub_best if ysub_stats else None,
        'ydom_detail': y_dom if y_dom_stats else None
    }
    return ysub_token, ydom_token, detail 