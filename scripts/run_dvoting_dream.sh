export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# "math500" "gpqa" "arc-c" "mmlu"
datasets=("gsm8k")
seq_lens=(256)
temperatures=(0.6)
sampling_upper_bound=(5)
entropy_thresh=(0.3)
block_sizes=(16)

for ds in "${datasets[@]}"; do
    for sl in "${seq_lens[@]}"; do
        for temperature in "${temperatures[@]}"; do
            for upper_bound in "${sampling_upper_bound[@]}"; do
                for thresh in "${entropy_thresh[@]}"; do
                    for block_size in "${block_sizes[@]}"; do
                        accelerate launch --main_process_port 18101 dream_distributed.py \
                        --dataset "$ds" \
                        --bsz 1 \
                        --alg entropy \
                        --seq-len "$sl" \
                        --block-size $block_size \
                        --steps 88 \
                        --temperature $temperature \
                        --top-p 0.9 \
                        --enable-remask-sampling \
                        --enable-adaptive-decoding \
                        --entropy-thresh $thresh \
                        --sampling-upper-bound $upper_bound
                    done
                done
            done
        done
    done
done

