python ./src/counseling/NQCP/train_NQCP.py \
--abuse_type 방임 \
--model_name klue/roberta-large \
--lr 1e-5 \
--batch_size 8 \
--epochs 10 \
--weight_decay 0.01 \
--seed 42

python ./src/counseling/NQCP/train_NQCP.py \
--abuse_type 정서학대 \
--model_name klue/bert-base \
--lr 1e-5 \
--batch_size 8 \
--epochs 10 \
--weight_decay 0.01 \
--seed 42

python ./src/counseling/NQCP/train_NQCP.py \
--abuse_type 신체학대 \
--model_name klue/roberta-large \
--lr 5e-6 \
--batch_size 8 \
--epochs 10 \
--weight_decay 0.01 \
--seed 42

python ./src/counseling/NQCP/train_NQCP.py \
--abuse_type 성학대 \
--model_name klue/roberta-large \
--lr 1e-5 \
--batch_size 16 \
--epochs 20 \
--weight_decay 0.01 \
--seed 42