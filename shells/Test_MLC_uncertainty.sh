CUDA_VISIBLE_DEVICES=1 python ./src/abuse_detection/uncertainty/main.py \
--ckpt_dir="outputs/MLC/LLM/Qwen2.5-3B-Instruct/checkpoint-1280" \
--save_dir="outputs/uncertainty/Qwen2.5-3B-Instruct/ckpt-1280" \
--test_data="./data/processed/finetuning/test.json" \
--max_tokens=7

python ./src/abuse_detection/uncertainty/threshold.py \
