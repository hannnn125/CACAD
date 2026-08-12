# labeled dataset 생성
python ./src/preprocessing/preprocess_raw.py

# finetuning dataset 생성
python ./src/preprocessing/gen_ft_dataset.py --example_num 2 --seed 42

# clustering 적용
python ./src/preprocessing/clustering/main.py

