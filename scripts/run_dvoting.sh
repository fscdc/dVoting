export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# "math500" "gpqa" "arc-c" "mmlu"
datasets=("gsm8k")
seq_lens=(256)
block_sizes=(16)
temperatures=(0.6)
sampling_upper_bound=(5)
entropy_thresh=(0.3)

# --model-name 'GSAI-ML/LLaDA-1.5' \
for ds in "${datasets[@]}"; do
    for sl in "${seq_lens[@]}"; do
        for bs in "${block_sizes[@]}"; do
            for temperature in "${temperatures[@]}"; do
                for upper_bound in "${sampling_upper_bound[@]}"; do
                    for thresh in "${entropy_thresh[@]}"; do
                        accelerate launch --main_process_port 18100 llada_distributed.py \
                        --dataset "$ds" \
                        --bsz 1 \
                        --sampling-alg low_confidence \
                        --seq-len "$sl" \
                        --block-size $bs \
                        --steps 88 \
                        --temperature $temperature \
                        --cfg 0.0 \
                        --enable-adaptive-decoding \
                        --entropy-thresh $thresh \
                        --enable-remask-sampling --sampling-upper-bound $upper_bound \
                        --origin
                    done
                done
            done
        done
    done
done
