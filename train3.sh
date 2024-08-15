nohup python train2.py \
    --save_dir "checkpoints/DuIE" \
    --train_path "data/DuIE/train.txt" \
    --dev_path "data/DuIE/dev.txt" \
    --img_log_dir "logs/" \
    --img_log_name "UIE Base" \
    --batch_size 16 \
    --max_seq_len 256 \
    --learning_rate 5e-5 \
    --num_train_epochs 500 \
    --logging_steps 30 \
    --valid_steps 100\
    --device cuda:1 2>&1 &


nohup python train2.py \
    --save_dir "checkpoints/DuIE2" \
    --train_path "data/DuIE/train.txt" \
    --dev_path "data/DuIE/dev.txt" \
    --img_log_dir "logs/" \
    --img_log_name "UIE Base" \
    --batch_size 16 \
    --max_seq_len 256 \
    --learning_rate 2e-3 \
    --num_train_epochs 500 \
    --logging_steps 30 \
    --valid_steps 100 \
    --device cuda:0 \
    --txt_log_name "tr_log.txt" > my_output.log 2>&1 &

nohup python train3.py \
    --save_dir "checkpoints/DuIE" \
    --train_path "data/DuIE/train.txt" \
    --dev_path "data/DuIE/dev.txt" \
    --img_log_dir "logs/" \
    --img_log_name "UIE Base" \
    --batch_size 16 \
    --max_seq_len 256 \
    --learning_rate 2e-5 \
    --num_train_epochs 500 \
    --logging_steps 30 \
    --valid_steps 100 \
    --device cuda:0 \
    --txt_log_name "tr_log_v16_3.txt" > my_output.log 2>&1 &

python train3.py     --save_dir "checkpoints/UIE-SIAIB-ALL--2"     --train_path "data/UIE-SIAIB/train.txt"     --dev_path "data/UIE-SIAIB/dev.txt"     --img_log_dir "logs/"     --img_log_name "UIE New_ALL--2"     --batch_size 200     --max_seq_len 256     --learning_rate 2e-5     --num_train_epochs 600     --logging_steps 30     --valid_steps 100     --device cuda:1     --txt_log_name "tr_log_all--2.txt"

python train3.py     --save_dir "checkpoints/UIE-SIAIB-ALL--2"     --train_path "data/UIE-SIAIB/train.txt"     --dev_path "data/UIE-SIAIB/dev.txt"     --img_log_dir "logs/"     --img_log_name "UIE New_ALL--2"     --batch_size 200     --max_seq_len 256     --learning_rate 2e-5     --num_train_epochs 600     --logging_steps 30     --valid_steps 100     --device cuda:1     --txt_log_name "tr_log_all--2.txt"

python train3.py     --save_dir "checkpoints/UIE-SIAIB-ALL--3"     --train_path "data/DuIE2/train.txt"     --dev_path "data/DuIE2/dev.txt"     --img_log_dir "logs/"     --img_log_name "UIE New_ALL--3"     --batch_size 200     --max_seq_len 256     --learning_rate 2e-5     --num_train_epochs 600     --logging_steps 30     --valid_steps 100     --device cuda:1     --txt_log_name "tr_log_all--3.txt"

nohup python train3.py     --save_dir "checkpoints/diakg2-ib"     --train_path "data/diakg/train.txt"     --dev_path "data/diakg/dev.txt"     --img_log_dir "logs/"     --img_log_name "UIE diakg2"     --batch_size 280     --max_seq_len 340     --learning_rate 2e-5     --num_train_epochs 350     --logging_steps 30     --valid_steps 120     --device cuda:0     --txt_log_name "diakg2-ib.txt"  > my_output.log 2>&1 &

python doccano.py \
    --doccano_file ./data/diakg/doccano_ext.json \
    --task_type ext \
    --save_dir ./data/diakg \
    --splits 0.8 0.2 0 \
    --negative_ratio 3