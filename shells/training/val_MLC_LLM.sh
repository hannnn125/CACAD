CUDA_VISIBLE_DEVICES=0 python ./src/detection/MLC/LLM/val.py \
--base_ckpt_dir="outputs/MLC/LLM/Qwen2.5-3B-Instruct" \
--base_save_dir="outputs/MLC/LLM/Qwen2.5-3B-Instruct/val" \
--val_data="./data/processed/finetuning/val.json" \
--max_new_tokens=7
