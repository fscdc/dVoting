import os
import json
from glob import glob
from tqdm import tqdm
from collections import Counter
import re


def majority_voting_from_files(input_dir: str, output_path: str, prefix: str = "origin_", suffix: str = "", dataset: str = 'gsm8k', func: callable = None):
    files = sorted(glob(os.path.join(input_dir, f"{prefix}*{suffix}.json")))
    if not files:
        raise FileNotFoundError(f"No {prefix}*{suffix}.json files found in {input_dir}")

    print(f"Found {len(files)} origin files:")
    for f in files:
        print(" -", os.path.basename(f))

    all_data = []
    for file in tqdm(files, desc="Loading files"):
        results = []
        with open(file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        all_data.append(results)

    num_files = len(all_data)
    num_samples = len(all_data[0])

    for i, data in enumerate(all_data):
        assert len(data) == num_samples, f"File {files[i]} has inconsistent sample count."

    print(f"\nPerforming majority voting on {num_samples} samples across {num_files} files...\n")

    output_data = []
    
    for idx in tqdm(range(num_samples), desc="Voting samples"):
        answers = []
        steps_total = 0
        meta = all_data[0][idx].copy()

        for file_idx in range(num_files):
            answer_list = all_data[file_idx][idx]['answer']
            steps_total += all_data[file_idx][idx].get('steps', 0)

            for answer in answer_list:
                if dataset in ['gsm8k', 'math500', 'arc-c', 'gpqa', 'mmlu']:
                    strict, flex = func(answer)
                else:
                    raise NotImplementedError

                if dataset == 'gsm8k' or dataset == 'arc-c':
                    if len(strict) > 0:
                        answers.append(strict[0])
                    else:
                        if flex:
                            answers.append(flex[0])

                elif dataset == 'math500':
                    from evaluation.math500.math_normalize import normalize_answer
                    from evaluation.math500.grader import _normalize
                    if strict is not None:
                        # strict = normalize_answer(strict)
                        strict = _normalize(strict)
                        answers.append(strict)
                    else:
                        if flex:
                            answers.append(flex)

                elif dataset in ['gpqa', 'mmlu']:
                    if strict is not None:
                        answers.append(strict)
                    else:
                        if flex:
                            answers.append(flex)

        if not answers:
            voted = "NoAnswer"
        else:
            count = Counter(answers)
            top = count.most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                voted = top[0][0]
            else:
                voted = top[0][0]

        meta['answer'] = [f"\\boxed{{{voted}}}"]
        meta['steps'] = steps_total
        output_data.append(meta)

    print(f"\nSaving majority-voted results to: {output_path}")
    with open(output_path, "w") as f:
        for sample in tqdm(output_data, desc="Writing output"):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("✅ Done!")

if __name__ == "__main__":
    # an example usage
    INPUT_DIR = "to-add/result_summary/gsm8k"
    OUTPUT_PATH = "to-add/result_summary/gsm8k/majority_voted_results.json"
    DATASET = "gsm8k"  # Options: gsm8k, math500, gpqa, mmlu

    # this is the prefix and suffix for match target files (we have to match multiple files (multiple results) for voting)
    PREFIX = "origin_len_128_steps_64_low_confidence_block_"
    SUFFIX = ""

    if DATASET == 'gsm8k':
        from evaluation.gsm8k.eval_gsm8k import evaluation, time_evaluation
        from evaluation.gsm8k.eval_gsm8k import extract_boxed_text
    elif DATASET == 'math500':
        from evaluation.math500.eval_math500 import evaluation, time_evaluation
        from evaluation.math500.eval_math500 import extract_boxed_text
    elif DATASET == 'gpqa':
        from evaluation.gpqa.eval_gpqa import evaluation, time_evaluation
        from evaluation.gpqa.eval_gpqa import extract_boxed_text_sampling
        extract_boxed_text = extract_boxed_text_sampling
    elif DATASET == 'mmlu':
        from evaluation.mmlu.eval_mmlu import evaluation, time_evaluation
        from evaluation.mmlu.eval_mmlu import extract_boxed_text_sampling # only one return value
        extract_boxed_text = extract_boxed_text_sampling
    elif DATASET == 'arc-c':
        from evaluation.arc.eval_arc import evaluation, time_evaluation
        from evaluation.arc.eval_arc import extract_boxed_text
    else:
        raise NotImplementedError

    
    majority_voting_from_files(INPUT_DIR, OUTPUT_PATH, PREFIX, SUFFIX, DATASET, extract_boxed_text)

    # evaluate the final results
    final_result = evaluation(OUTPUT_PATH)
    print("\nFinal Evaluation Results:")
    for k, v in final_result.items():
        print(f"{k}: {v}")

    # save results
    # result_save_path = OUTPUT_PATH.replace("majority_voted_results.json", "majority_voted_results_evaluated.json")
    # with open(result_save_path, "w") as f:
    #     json.dump(final_result, f, indent=4)

    print("Done!")