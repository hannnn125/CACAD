python ./src/counseling/offensive_question/train_offensive_detection.py \
--models klue/bert-base klue/roberta-large BM-K/KoSimCSE-roberta \
--epoch 15 \
--batch_size 16 \
--lr 1e-5 \
--weight_decay 0.01 \
--max_length 512 \
--data_dir data/processed/offensive_dataset \
--output_dir outputs \
--cache_dir ./.cache/huggingface