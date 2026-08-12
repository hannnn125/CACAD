CUDA_VISIBLE_DEVICES=1 python ./src/detection/MLC/LLM/train.py \
--cache_dir='/nas/user10/.cache/yg/' \
--config_name="Qwen/Qwen2.5-3B-Instruct" \
--tokenizer_name="Qwen/Qwen2.5-3B-Instruct" \
--model_name_or_path="Qwen/Qwen2.5-3B-Instruct" \
--train_file="./data/processed/finetuning/train.json" \
--num_train_epochs=3 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=2 \
--alpha=0.1 \
--output_dir='/nas/counsell/log/test' \
--do_train \
--fp16=True \
--save_strategy='epoch' \
--eval_strategy='epoch' \
--save_total_limit=3 \
--load_best_model_at_end=True \
--metric_for_best_model='eval_loss' \
--greater_is_better=False \
--optim='adafactor' \
--learning_rate=1e-5 \
--logging_strategy='steps' \
--logging_first_step \
--report_to='none' \
--do_eval=True \
--validation_file="./data/processed/finetuning/val.json" \
--max_eval_samples=10 \
--logging_dir='/nas/counsell/log/test/runs' \
--run_name='260520_Qwen2.5-3B-Instruct_base_ep(3)_lr(1e-5)_batch(2)_alpha(0.1)' \
--low_cpu_mem_usage \
--remove_unused_columns false \
--seed=42

# CUDA_VISIBLE_DEVICES=1 python ./src/detection/MLC/LLM/train.py \
# --cache_dir='/nas/user10/.cache/yg/' \
# --config_name="polyglot/Polyglot-3.5-3B-Instruct" \
# --tokenizer_name="polyglot/Polyglot-3.5-3B-Instruct" \
# --model_name_or_path="polyglot/Polyglot-3.5-3B-Instruct" \
# --train_file="./data/processed/finetuning/train.json" \
# --num_train_epochs=3 \
# --per_device_train_batch_size=2 \
# --gradient_accumulation_steps=2 \
# --alpha=0.1 \
# --output_dir='/nas/counsell/log/test' \
# --do_train \
# --fp16=True \
# --save_strategy='epoch' \
# --eval_strategy='epoch' \
# --save_total_limit=3 \
# --load_best_model_at_end=True \
# --metric_for_best_model='eval_loss' \
# --greater_is_better=False \
# --optim='adafactor' \
# --learning_rate=1e-5 \
# --logging_strategy='steps' \
# --logging_first_step \
# --report_to='none' \
# --do_eval=True \
# --validation_file="./data/processed/finetuning/val.json" \
# --max_eval_samples=10 \
# --logging_dir='/nas/counsell/log/test/runs' \
# --run_name='260520_Polyglot-3.5-3B-Instruct_base_ep(3)_lr(1e-5)_batch(2)_alpha(0.1)' \
# --low_cpu_mem_usage \
# --remove_unused_columns false \
# --seed=42

