CUDA_VISIBLE_DEVICES=0 python ./src/detection/MLC/LLM/val.py \
--base_ckpt_dir="/nas/counsell/log/test/" \
--base_save_dir="./src/detection/MLC/LLM/output/val" \
--val_data="./data/processed/finetuning/val.json" \
--max_new_tokens=7
