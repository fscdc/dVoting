export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# "math500" "gpqa" "arc-c" "mmlu"
datasets=("gsm8k") 
seq_lens=(256)
steps=(128)
block_sizes=(8 16 32 64 128)

# seq_lens=(128)
# steps=(64)
# block_sizes=(4 8 16 32 64)

# seq_lens=(512)
# steps=(256)
# block_sizes=(16 32 64 128 256)
seed=(42 43 44 45 46)

# --model-name 'GSAI-ML/LLaDA-1.5' \
for ds in "${datasets[@]}"; do
    for sl in "${seq_lens[@]}"; do
        for bs in "${block_sizes[@]}"; do
            for st in "${steps[@]}"; do
                for sd in "${seed[@]}"; do
                    accelerate launch --main_process_port 18100 llada_distributed.py \
                        --dataset "$ds" \
                        --bsz 1 \
                        --sampling-alg low_confidence \
                        --seq-len "$sl" \
                        --block-size $bs \
                        --steps $st \
                        --temperature 0.9 \
                        --cfg 0.0 \
                        --seed $sd \
                        --origin     
                done
            done
        done
    done
done
