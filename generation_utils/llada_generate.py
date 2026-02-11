import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, enable_record=False, **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # @sicheng: for record step traces
    B = x.size(0)
    gen_start = prompt.shape[1]
    gen_end   = gen_start + gen_length

    records = [] if enable_record else None
    
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            global_step = num_block * steps + i

            if enable_record:
                step_record = {
                    "global_step": global_step,
                    "entropy": None,
                    "confidence": None,
                    "commit_pos": [],
                }

            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # greedy decoding (temperature = 0)
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l


            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                if enable_record:
                    entropy = -(p * (p.clamp(min=1e-12)).log()).sum(dim=-1)
                    entropy_vector = entropy[0, gen_start:gen_end].detach().cpu().tolist()
                    confidence_vector = (-entropy)[0, gen_start:gen_end].detach().cpu().tolist()
                    x0_p_origin = x0_p.clone()
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                if enable_record:
                    raise NotImplementedError("Random remasking does not support record yet.")
            else:
                raise NotImplementedError(remasking)

            # @sicheng: actually fake remasking (semi-AR)
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            selected_pos_per_batch = []

            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                if enable_record:
                    for pos in select_index.tolist():
                        if gen_start <= pos < gen_end:
                            step_record["commit_pos"].append(int(pos - gen_start))

            x[transfer_index] = x0[transfer_index]

            if enable_record:
                step_record["entropy"] = entropy_vector
                step_record["confidence"] = confidence_vector
                records.append(step_record)
    
    if enable_record:
        return x, records

    return x




from generation_utils.remask_utils import answer_level_consistency

@torch.no_grad()
def generate_entropy_adaptive(model, tokenizer, prompt, steps=128, gen_length=128, block_length=128, temperature=0., cfg_scale=0.,
    remasking='low_confidence', mask_id=126336, entropy_thresh=2.5, enable_remask_sampling=False, sampling_upper_bound=5, dataset="gsm8k", enable_record=False, **kwargs):
    """
    Adaptive multi-token commit decoding based on entropy threshold.

    Args:
        model: Mask predictor (dLLM).
        tokenizer: Tokenizer for decoding (unused but kept for compatibility).
        prompt: Tensor of shape (1, L).
        steps: Maximum sampling steps.
        gen_length: Length of generation.
        block_length: Length per block.
        temperature: Sampling temperature.
        cfg_scale: CFG scale.
        remasking: Remasking strategy: 'low_confidence' or 'random'.
        mask_id: Token id of [MASK].
        entropy_thresh: float. Tokens with entropy < threshold (i.e., confident tokens) are committed.
    Returns:
        x: final generated tensor
        actual_steps: int, number of adaptive steps executed
    """
    seed_list = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
    if enable_remask_sampling and enable_record == False:
        x_final = None
        total_steps = 0
        i = 0

        while (i < sampling_upper_bound):

            # @sicheng: global-level consistency adaptive sampling
            if i > 1:
                x_final_copy = x_final.clone()
                x_final_copy = x_final_copy[:, prompt.shape[1]:].contiguous()
                result_list = tokenizer.batch_decode(x_final_copy, skip_special_tokens=True)

                top_list = answer_level_consistency(result_list, dataset)

                answer_sum = sum(top_list)

                if answer_sum <= 0:
                    pass # @sicheng: if no valid answer, continue sampling
                else:
                    ratio = top_list[0] / answer_sum

                    if top_list[0] > int(sampling_upper_bound / 2) or (ratio == 1.0 and top_list[0] >= int(sampling_upper_bound / 2)):
                        break

            # Prepare@sicheng: set diff seed
            set_random_seed(seed_list[i])

            i += 1
            B = prompt.shape[0]
            device = model.device

            def func_remask(x_all, x, start_idx, sampling_upper_bound, sampling_iter, mask_id, consistency_thresh=1.0):
                new_x = x.clone()
                B, L = new_x.shape
                assert B == 1, "Remask sampling only supports batch size 1."

                gen_len = L - start_idx
                assert gen_len > 0, "start_idx must be within sequence length."

            
                if sampling_iter <= int((sampling_upper_bound / 2) + 1):
                    new_x[:, start_idx:] = mask_id
                    return new_x
                
                past = x_all[:, start_idx:].clone()    # shape: N x gen_len
                T = past.shape[0]

                for idx in range(gen_len):
                    tokens = past[:, idx]
                    values, counts = torch.unique(tokens, return_counts=True)
                    top_idx = torch.argmax(counts).item()
                    top_token = values[top_idx]
                    ratio = counts[top_idx].item() / T

                    if ratio >= consistency_thresh:
                        new_x[:, start_idx + idx] = top_token
                    else:
                        new_x[:, start_idx + idx] = mask_id

                # @sicheng: always keep last quarter masked
                quarter_len = gen_len // 4
                new_x[:, L - quarter_len:] = mask_id

                return new_x


            if x_final is None:
                x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
                x[:, :prompt.shape[1]] = prompt.clone()
                prompt_index = (x != mask_id)
            else:
                # @sicheng: token-level consistency adaptive remasking 
                x_final_copy = x_final.clone()
                x = func_remask(x_final_copy, x, prompt.shape[1], sampling_upper_bound, i, mask_id, 1.0).clone()

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            gen_start = prompt.shape[1]
            gen_end = gen_start + gen_length

            def all_done():
                return torch.all(x[:, gen_start:gen_end] != mask_id)

            while not all_done():
                mask_index = (x == mask_id)

                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

                # === compute entropy ===
                p = F.softmax(logits.to(torch.float64), dim=-1)
                log_p = torch.log(p + 1e-12)
                entropy = -(p * log_p).sum(dim=-1)  # (B, L)
                confidence = -entropy  # higher is better

                # focus only on masked tokens
                confidence = torch.where(mask_index, confidence, -np.inf)

                commit_guard = True

                # === block-wise commit ===
                for blk in range(num_blocks):
                    blk_start = gen_start + blk * block_length
                    blk_end = blk_start + block_length

                    blk_mask = mask_index[:, blk_start:blk_end]
                    blk_entropy = entropy[:, blk_start:blk_end]
                    blk_conf = confidence[:, blk_start:blk_end]
                    blk_x0 = x0[:, blk_start:blk_end]

                    if not blk_mask.any():
                        continue  # already completed

                    # ensure left-to-right block dependency
                    left_blocks_done = True
                    for prev_blk in range(blk):
                        prev_start = gen_start + prev_blk * block_length
                        prev_end = prev_start + block_length
                        if (x[:, prev_start:prev_end] == mask_id).any():
                            left_blocks_done = False
                            break
                    if not left_blocks_done:
                        break

                    # === select tokens with entropy below threshold ===
                    select_mask = (blk_entropy < entropy_thresh) & blk_mask # @sicheng: important!

                    if (not select_mask.any()) and commit_guard:
                        # === min commit: at least commit one token ===
                        safe_conf = torch.where(blk_mask, blk_conf, -torch.inf)
                        max_conf_idx = torch.argmax(safe_conf[0])
                        commit_range = slice(blk_start + max_conf_idx, blk_start + max_conf_idx + 1)
                        x[:, commit_range] = x0[:, commit_range]
                        commit_guard = False
                        continue

                    # commit all confident tokens (non-contiguous allowed)
                    select_indices = torch.nonzero(select_mask[0], as_tuple=False).squeeze(-1)
                    x[:, blk_start + select_indices] = x0[:, blk_start + select_indices]

                    # if current block not done, stop proceeding to right blocks
                    if (x[:, blk_start:blk_end] == mask_id).any():
                        break

                total_steps += 1

                
            if x_final is None:
                x_final = x.clone()
            else:
                x_final = torch.cat([x_final, x.clone()], dim=0)
            
        return x_final, total_steps
    elif enable_remask_sampling and enable_record == True: # more progressive dTTS
        records = [] if enable_record else None

        x_final = None
        total_steps = 0
        i = 0

        while (i < sampling_upper_bound):

            # @sicheng: global-level consistency adaptive sampling
            if i > 1:
                x_final_copy = x_final.clone()
                x_final_copy = x_final_copy[:, prompt.shape[1]:].contiguous()
                result_list = tokenizer.batch_decode(x_final_copy, skip_special_tokens=True)

                top_list = answer_level_consistency(result_list, dataset)

                answer_sum = sum(top_list)

                if answer_sum <= 0:
                    pass # @sicheng: if no valid answer, continue sampling
                else:
                    ratio = top_list[0] / answer_sum

                    if top_list[0] > int(sampling_upper_bound / 2) or (ratio == 1.0 and top_list[0] >= int(sampling_upper_bound / 2)):
                        break

            # Prepare@sicheng: set diff seed
            set_random_seed(seed_list[i])

            i += 1
            B = prompt.shape[0]
            device = model.device

            def func_remask(x_all, x, start_idx, sampling_upper_bound, sampling_iter, mask_id, consistency_thresh=1.0):
                new_x = x.clone()
                B, L = new_x.shape
                assert B == 1, "Remask sampling only supports batch size 1."

                gen_len = L - start_idx
                assert gen_len > 0, "start_idx must be within sequence length."

            
                if sampling_iter <= int((sampling_upper_bound / 2) + 1):
                    new_x[:, start_idx:] = mask_id
                    return new_x
                
                past = x_all[:, start_idx:].clone()    # shape: N x gen_len
                T = past.shape[0]

                for idx in range(gen_len):
                    tokens = past[:, idx]
                    values, counts = torch.unique(tokens, return_counts=True)
                    top_idx = torch.argmax(counts).item()
                    top_token = values[top_idx]
                    ratio = counts[top_idx].item() / T

                    if ratio >= consistency_thresh:
                        new_x[:, start_idx + idx] = top_token
                    else:
                        new_x[:, start_idx + idx] = mask_id

                # @sicheng: always keep last quarter masked
                quarter_len = gen_len // 4
                new_x[:, L - quarter_len:] = mask_id

                return new_x


            if x_final is None:
                x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
                x[:, :prompt.shape[1]] = prompt.clone()
                prompt_index = (x != mask_id)
            else:
                # @sicheng: token-level consistency adaptive remasking 
                x_final_copy = x_final.clone()
                x = func_remask(x_final_copy, x, prompt.shape[1], sampling_upper_bound, i, mask_id, 1.0).clone()

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            gen_start = prompt.shape[1]
            gen_end = gen_start + gen_length

            def all_done():
                return torch.all(x[:, gen_start:gen_end] != mask_id)

            while not all_done():
                mask_index = (x == mask_id)

                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

                # === compute entropy ===
                p = F.softmax(logits.to(torch.float64), dim=-1)
                log_p = torch.log(p + 1e-12)
                entropy = -(p * log_p).sum(dim=-1)  # (B, L)
                confidence = -entropy  # higher is better

                # focus only on masked tokens
                confidence = torch.where(mask_index, confidence, -np.inf)

                commit_guard = True

                # === block-wise commit ===
                for blk in range(num_blocks):
                    blk_start = gen_start + blk * block_length
                    blk_end = blk_start + block_length

                    blk_mask = mask_index[:, blk_start:blk_end]
                    blk_entropy = entropy[:, blk_start:blk_end]
                    blk_conf = confidence[:, blk_start:blk_end]
                    blk_x0 = x0[:, blk_start:blk_end]

                    if not blk_mask.any():
                        continue  # already completed

                    # ensure left-to-right block dependency
                    left_blocks_done = True
                    for prev_blk in range(blk):
                        prev_start = gen_start + prev_blk * block_length
                        prev_end = prev_start + block_length
                        if (x[:, prev_start:prev_end] == mask_id).any():
                            left_blocks_done = False
                            break
                    if not left_blocks_done:
                        break

                    # === select tokens with entropy below threshold ===
                    select_mask = (blk_entropy < entropy_thresh) & blk_mask # @sicheng: important!

                    if (not select_mask.any()) and commit_guard:
                        # === min commit: at least commit one token ===
                        safe_conf = torch.where(blk_mask, blk_conf, -torch.inf)
                        max_conf_idx = torch.argmax(safe_conf[0])
                        commit_range = slice(blk_start + max_conf_idx, blk_start + max_conf_idx + 1)
                        x[:, commit_range] = x0[:, commit_range]
                        commit_guard = False
                        continue

                    # commit all confident tokens (non-contiguous allowed)
                    select_indices = torch.nonzero(select_mask[0], as_tuple=False).squeeze(-1)
                    x[:, blk_start + select_indices] = x0[:, blk_start + select_indices]

                    # if current block not done, stop proceeding to right blocks
                    if (x[:, blk_start:blk_end] == mask_id).any():
                        break

                total_steps += 1

                
            if x_final is None:
                x_final = x.clone()
            else:
                x_final = torch.cat([x_final, x.clone()], dim=0)
            
        return x_final, total_steps
    else:
        records = [] if enable_record else None

        B = prompt.shape[0]
        device = model.device

        x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :prompt.shape[1]] = prompt.clone()
        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        total_steps = 0
        gen_start = prompt.shape[1]
        gen_end = gen_start + gen_length

        def all_done():
            return torch.all(x[:, gen_start:gen_end] != mask_id)

        while not all_done():
            if enable_record:
                step_record = {
                    "global_step": total_steps,
                    "entropy": None,
                    "confidence": None,
                    "commit_pos": [],
                }

            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

            # === compute entropy ===
            p = F.softmax(logits.to(torch.float64), dim=-1)
            log_p = torch.log(p + 1e-12)
            entropy = -(p * log_p).sum(dim=-1)  # (B, L)
            confidence = -entropy  # higher is better

            if enable_record:
                step_record["entropy"] = (
                    entropy[:, gen_start:gen_end]
                    .detach()
                    .cpu()
                    .tolist()
                )
                step_record["confidence"] = (
                    confidence[:, gen_start:gen_end]
                    .detach()
                    .cpu()
                    .tolist()
                )

            # focus only on masked tokens
            confidence = torch.where(mask_index, confidence, -np.inf)

            commit_guard = True

            # === block-wise commit ===
            for blk in range(num_blocks):
                blk_start = gen_start + blk * block_length
                blk_end = blk_start + block_length

                blk_mask = mask_index[:, blk_start:blk_end]
                blk_entropy = entropy[:, blk_start:blk_end]
                blk_conf = confidence[:, blk_start:blk_end]
                blk_x0 = x0[:, blk_start:blk_end]

                if not blk_mask.any():
                    continue  # already completed

                # ensure left-to-right block dependency
                left_blocks_done = True
                for prev_blk in range(blk):
                    prev_start = gen_start + prev_blk * block_length
                    prev_end = prev_start + block_length
                    if (x[:, prev_start:prev_end] == mask_id).any():
                        left_blocks_done = False
                        break
                if not left_blocks_done:
                    break

                # === select tokens with entropy below threshold ===
                select_mask = (blk_entropy < entropy_thresh) & blk_mask # @sicheng: important!

                if (not select_mask.any()) and commit_guard:
                    # === min commit: at least commit one token ===
                    safe_conf = torch.where(blk_mask, blk_conf, -torch.inf)
                    max_conf_idx = torch.argmax(safe_conf[0])
                    commit_range = slice(blk_start + max_conf_idx, blk_start + max_conf_idx + 1)
                    x[:, commit_range] = x0[:, commit_range]

                    if enable_record:
                        step_record["commit_pos"].append(int(blk_start + max_conf_idx))

                    commit_guard = False
                    continue

                # commit all confident tokens (non-contiguous allowed)
                select_indices = torch.nonzero(select_mask[0], as_tuple=False).squeeze(-1)
                x[:, blk_start + select_indices] = x0[:, blk_start + select_indices]

                if enable_record:
                    commit_positions = blk_start + select_indices
                    step_record["commit_pos"].extend(
                        commit_positions.tolist()
                    )

                # if current block not done, stop proceeding to right blocks
                if (x[:, blk_start:blk_end] == mask_id).any():
                    break

            if enable_record:
                step_record["commit_pos"] = [
                    int(pos - gen_start) for pos in step_record["commit_pos"]
                ]
                records.append(step_record)

            total_steps += 1

            # if total_steps >= steps:
            #     break
        
        if enable_record:
            return x, total_steps, records

        return x, total_steps

