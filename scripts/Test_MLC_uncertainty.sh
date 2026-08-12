CUDA_VISIBLE_DEVICES=1 python ./src/detection/uncertainty/src/main.py \
--ckpt_dir="/nas/counsell/log/test/checkpoint-1280" \
--base_save_dir="./src/detection/uncertainty/output/test/ckpt-1280" \
--test_data="./data/processed/finetuning/test.json" \
--max_tokens=7