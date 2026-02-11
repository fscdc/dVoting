export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# "math500" "gpqa" "arc-c" "mmlu"
datasets=("gsm8k")
seq_lens=(128)
steps=(64)
block_sizes=(8)

# seq_lens=(256)
# steps=(128)
# block_sizes=(16)

seed=(42 43 44 45 46)
topp_values=(0.6)
temperatures=(0.9)

for ds in "${datasets[@]}"; do
    for sl in "${seq_lens[@]}"; do
        for st in "${steps[@]}"; do
            for sd in "${seed[@]}"; do
                for tp in "${topp_values[@]}"; do
                    for temp in "${temperatures[@]}"; do
                        for bs in "${block_sizes[@]}"; do
                            accelerate launch --main_process_port 18100 dream_distributed.py \
                                --dataset "$ds" \
                                --bsz 1 \
                                --alg entropy \
                                --seq-len "$sl" \
                                --block-size $bs \
                                --steps $st \
                                --temperature $temp \
                                --top-p $tp \
                                --seed $sd \
                                --semiar-origin     
                        done
                    done 
                done 
            done
        done
    done
done

