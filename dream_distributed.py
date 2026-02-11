import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer

import os
import time
import datetime
import argparse
from tqdm import tqdm

import json

import torch
import torch.distributed as dist

import time


def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dataloader_by_rank(ds, bsz, rank, world_size):
    index = 0
    while index * bsz * world_size < len(ds):
        start_idx = index * bsz * world_size + bsz * rank
        if start_idx >= len(ds):
            yield []
        elif start_idx + bsz >= len(ds):
            yield ds.select(range(start_idx, len(ds)))
        else:
            yield ds.select(range(start_idx,start_idx+bsz))

        index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--sampling-alg", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--evaluation-only", action="store_true", default=False)
    parser.add_argument("--time-evaluation", action="store_true", default=False)
    parser.add_argument("--subset", action="store_true", default=False)

    parser.add_argument("--origin", action="store_true", help="use origin")
    parser.add_argument("--prefill", action="store_true", help="use prefill")
    parser.add_argument("--kv-cache-greedy", action="store_true", help="use kv cache greedy")
    parser.add_argument("--kv-cache-decoded", action="store_true", help="use kv cache decoded")
    parser.add_argument("--kv-cache-masked", action="store_true", help="use kv cache masked")
    parser.add_argument("--cache-steps", type=int, default=0, help="number of steps to cache")

    parser.add_argument("--alg", type=str, default='entropy')

    # @sicheng: add
    parser.add_argument("--model-name", type=str, default='Dream-org/Dream-v0-Instruct-7B')
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--enable-adaptive-decoding", action="store_true", default=False)
    parser.add_argument("--entropy-thresh", type=float, default=0.5)
    parser.add_argument("--enable-remask-sampling", action="store_true", default=False)
    parser.add_argument("--sampling-upper-bound", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--semiar-origin", action="store_true", default=False)

    args = parser.parse_args()


    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=36000))
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    device = 'cuda:{}'.format(rank)


    if args.kv_cache_decoded or args.origin:
        from models.modeling_dream import DreamModel
    elif args.enable_remask_sampling and args.enable_remask_sampling:
        from models.modeling_dream_remask_sampling import DreamModel
    elif args.semiar_origin:
        from models.modeling_dream_semiar_origin import DreamModel
    elif args.kv_cache_greedy:
        from models.modeling_dream_greedy import DreamModel
    else:
        from models.modeling_dream_pd import DreamModel

    model = DreamModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    from datasets import load_dataset
    if args.dataset == 'gsm8k':
        from evaluation.gsm8k.prompt import format_prompt
        from evaluation.gsm8k.eval_gsm8k import evaluation, time_evaluation
        ds = load_dataset('openai/gsm8k', 'main')['test'] # test
    elif args.dataset == 'math500':
        from evaluation.math500.prompt import format_prompt
        from evaluation.math500.eval_math500 import evaluation, time_evaluation
        ds = load_dataset("HuggingFaceH4/MATH-500")['test']
    elif args.dataset == 'gpqa': # Use simple-eval
        from evaluation.gpqa.prompt import format_prompt
        from evaluation.gpqa.eval_gpqa import evaluation, time_evaluation
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main")['train']
    elif args.dataset == 'humaneval':  # Use simple-eval
        from evaluation.humaneval.prompt import format_prompt
        from evaluation.humaneval.eval_human_eval import time_evaluation
        ds = load_dataset('openai/openai_humaneval')['test'] # test
    elif args.dataset == 'mmlu':
        from evaluation.mmlu.prompt import format_prompt
        from evaluation.mmlu.eval_mmlu import evaluation, time_evaluation
        ds = load_dataset('cais/mmlu', 'all')['test']
    elif args.dataset == 'mbpp':
        from evaluation.mbpp.prompt import format_prompt
        from evaluation.mbpp.eval_mbpp import evaluation, time_evaluation
        ds = load_dataset('google-research-datasets/mbpp')['test']
    elif args.dataset == 'deepscaler':
        from evaluation.deepscaler.prompt import format_prompt
        from evaluation.deepscaler.eval_deepscaler import evaluation, time_evaluation
        ds = load_dataset('agentica-org/DeepScaleR-Preview-Dataset')['train']
    elif args.dataset == 'arc-c':
        from evaluation.arc.prompt import format_prompt
        from evaluation.arc.eval_arc import evaluation, time_evaluation
        ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge')['test']
    else:
        raise NotImplementedError
    results = []



    model = model.to(rank)
    model.eval()


    set_random_seed(args.seed)


    if args.time_evaluation:
        if args.subset:
            root_dir = 'dream_speed_subset'
        else:
            root_dir = 'dream_final'
    else:
        root_dir = 'result_summary_dream'

    if args.enable_remask_sampling and args.enable_remask_sampling:
        root_dir = root_dir + '_remask_sampling'
    elif args.semiar_origin:
        root_dir = root_dir + '_semiar'


    dir_name = f'{root_dir}/{args.dataset}'
    os.makedirs(dir_name+'/subprocess', exist_ok=True)
    if args.origin:
        filename = f"{dir_name}/subprocess/origin_len_{args.seq_len}_steps_{args.steps}_alg_{args.alg}_seed_{args.seed}_temp_{args.temperature}_topp_{args.top_p}"
    if args.semiar_origin:
        filename = f"{dir_name}/subprocess/semiar_origin_len_{args.seq_len}_steps_{args.steps}_block_size_{args.block_size}_alg_{args.alg}_seed_{args.seed}_temp_{args.temperature}_topp_{args.top_p}"
    elif args.enable_remask_sampling and args.enable_remask_sampling:
        filename = f"{dir_name}/subprocess/remask_len_{args.seq_len}_steps_{args.steps}_block_size_{args.block_size}_alg_{args.alg}_seed_{args.seed}_temp_{args.temperature}_topp_{args.top_p}_ethresh_{args.entropy_thresh}_supb_{args.sampling_upper_bound}"
    else:
        raise NotImplementedError

    if not args.evaluation_only:
        f = open(f'{filename}_rank_{rank}.json', 'w') 

    # @sicheng: for test
    if args.subset:
        ds = ds.select(range(16))
    
    ds = ds.add_column("index", list(range(len(ds))))
    for sample in tqdm(dataloader_by_rank(ds, args.bsz, rank, world_size), total=len(ds) // (world_size * args.bsz)+1):
        
        if args.evaluation_only:
            break

        m = []
        if len(sample) == 0:
            break

        for p in sample:
            question, answer = format_prompt(p)
            m.append(
                [{
                    "role": "user", 
                    "content": question
                }] 
            )

        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        bsz = len(m)

        input_prompt = tokenizer(
            prompt,
            padding_side = 'left', padding = 'longest',
            return_tensors = 'pt'
        )
        input_ids = input_prompt['input_ids'].to(rank)#.unsqueeze(0)#.repeat(bsz, 1)
        attention_mask = input_prompt['attention_mask'].to(rank)

        start_time = time.time()

        out = None
        if args.origin:
            out = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.seq_len,
                output_history=True,
                return_dict_in_generate=True,
                steps=args.steps,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.alg,
                alg_temp=0.,
                use_cache=False if args.origin else True, #args.kv_cache_decoded or args.kv_cache_masked or args.prefill,
                cache_type="greedy",# if args.kv_cache_decoded else "masked" if args.kv_cache_masked else "prefill",
                cache_steps=args.cache_steps,
                shift_type="left"
            )
        elif args.semiar_origin:
            out = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.seq_len,
                output_history=True,
                return_dict_in_generate=True,
                steps=args.steps,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.alg,
                alg_temp=0.,
                use_cache=False,
                block_size=args.block_size
            )
        elif args.enable_remask_sampling and args.enable_remask_sampling:
            out, steps = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.seq_len,
                output_history=True,
                return_dict_in_generate=True,
                steps=args.steps,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.alg,
                alg_temp=0.,
                use_cache=False,
                entropy_thresh=args.entropy_thresh,
                sampling_upper_bound=args.sampling_upper_bound,
                block_size=args.block_size
            )
        
        end_time = time.time()
        # print(out.sequences)

        out = out.sequences[:, input_ids.shape[1]:].contiguous()
        
        result = tokenizer.batch_decode(out, skip_special_tokens=True)

        for i, p in enumerate(sample):
            useful_token = (out[i] == 151643).nonzero(as_tuple=True)[0].sort().values
            if len(useful_token) > 0:
                useful_token = useful_token[0].item()
            else:
                useful_token = out.shape[1]

            idx = p['index']

            res = {
                    "answer": result,
                    "task": p,
                    "time": (end_time - start_time) / len(sample),
                    "tokens": out.shape[1],
                    "useful_tokens": useful_token,
                    "answer_index": answer,
                    "steps": steps if args.enable_remask_sampling and args.enable_remask_sampling else -1
                }

            f.write(json.dumps(res) + '\n')
            f.flush()
    
    if not args.evaluation_only:
        f.close()
    dist.barrier()

    if rank == 0:
        name = filename.split('/')[-1]
        all_f_name = f"{dir_name}/{name}" + '_all.json'
        all_f = open(all_f_name, 'w')
        for i in range(world_size):
            f = open(filename + f'_rank_{i}.json', 'r')
            for line in f:
                all_f.write(line)
            f.close()

        all_f.close()
        
        if args.time_evaluation:
            final_result = time_evaluation(all_f_name)
            final_result['config'] = args.__dict__
            result_path = f"{dir_name}/time_{name}.json"
        else:
            final_result = evaluation(all_f_name)
            final_result['config'] = args.__dict__
            result_path = f"{dir_name}/result_{name}.json"

        with open(result_path + '.json', 'w') as f:
            json.dump(final_result, f, indent=4)     

        print(final_result)


if __name__ == '__main__':
    main()