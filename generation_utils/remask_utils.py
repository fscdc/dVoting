from collections import Counter


def answer_level_consistency(answer_list, dataset):
    if dataset == 'gsm8k':
        from evaluation.gsm8k.eval_gsm8k import extract_boxed_text
        func = extract_boxed_text
    elif dataset == 'math500':
        from evaluation.math500.eval_math500 import extract_boxed_text
        func = extract_boxed_text
    elif dataset == 'gpqa':
        from evaluation.gpqa.eval_gpqa import extract_boxed_text_sampling
        func = extract_boxed_text_sampling
    elif dataset == 'mmlu':
        from evaluation.mmlu.eval_mmlu import extract_boxed_text_sampling
        func = extract_boxed_text_sampling
    elif dataset == 'arc-c':
        from evaluation.arc.eval_arc import extract_boxed_text
        func = extract_boxed_text

    answers = []

    for answer in answer_list:
        if dataset in ['gsm8k', 'math500', 'arc-c', 'gpqa', 'mmlu']:
            strict, flex = func(answer)
        # elif dataset in []:
        #     flex = func(answer)
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

    count = Counter(answers)
    counts_sorted = [c for (_, c) in count.most_common()]

    return counts_sorted

if __name__ == "__main__":
    answer_list = [
        "\\boxed{24}",
        "\\boxed{24}",
        "\\boxed{26}",
        "\\boxed{24}",
        "\\boxed{25}",
        "\\boxed{24}",
        "\\boxed{27}",
    ]

    dataset = 'gsm8k'
    counts_sorted = answer_level_consistency(answer_list, dataset)
    print(counts_sorted)  # [4, 1, 1, 1]