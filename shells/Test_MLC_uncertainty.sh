CUDA_VISIBLE_DEVICES=1 python ./src/abuse_detection/uncertainty/main.py \
--ckpt_dir="/nas/user10_1/Counsell/log/Qwen2.5/250413_Qwen/Qwen2.5-3B_base_ep(20)_lr(1e-5)_batch(2)_alpha(0.1)/checkpoint-1280" \
--save_dir="./outputs/uncertainty/QWEN2.5-3B_1e-5/ckpt-1280" \
--test_data="./data/processed/finetuning/test.json" \
--max_tokens=7

python ./src/abuse_detection/uncertainty/threshold.py \

# CUDA_VISIBLE_DEVICES=1 python ./src/abuse_detection/uncertainty/main.py \
# --ckpt_dir="/nas/user10_1/Counsell/log/polyglot-5.8b/250413_lr(1e-5)_batch(4)_alpha(0.1)/checkpoint-5120" \
# --save_dir="./outputs/uncertainty/polyglot-ko-5.8b/ckpt-5120" \
# --test_data="./data/processed/finetuning/test.json" \
# --max_tokens=4

# python ./src/abuse_detection/uncertainty/threshold.py
