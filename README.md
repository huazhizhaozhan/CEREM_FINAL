# CEREM

## Requirements

Please ensure the following dependencies are installed：

- `torch`
- `transformers==4.22.1`
- `datasets==2.4.0`
- `evaluate==3.5.0`
- `matplotlib==3.5.0`
- `rich==12.5.1`
- `scikit-learn==1.1.2`
- `requests==2.28.1`
- `jieba`
- `tqdm`

## File Structure

1. Create a folder named `pretrain`, and place the pre-trained model `chinese-roberta-wwm-ext` in it.
2. Create a folder named `data`, which will be used to store the dataset files (`dev.txt`, `text.txt`, and `train.txt`).

## Method to Obtain the Dataset

### 1. Dataset constructed in this paper:
The link will be provided in the future.

### 2. Obtain the dataset by yourself:
Download and install `doccano` for data labeling using the "sequence labeling" task, then run the `doccano.py` script:

```bash
python doccano.py \
    --doccano_file ./data/diakg/doccano_ext.json \
    --task_type ext \
    --save_dir ./data/diakg \
    --splits 0.8 0.2 0 \
    --negative_ratio 3
```
## Training
Run `train3.py` to train the model. Note that you should modify the `--device_list` parameter in `train3.py` to match your environment.

```bash
python train3.py \
    --save_dir "checkpoints/diakg2-ib" \
    --train_path "data/diakg/train.txt" \
    --dev_path "data/diakg/dev.txt" \
    --img_log_dir "logs/" \
    --img_log_name "UIE diakg2" \
    --batch_size 280 \
    --max_seq_len 340 \
    --learning_rate 2e-5 \
    --num_train_epochs 350 \
    --logging_steps 30 \
    --valid_steps 120 \
    --device cuda:0 \
    --txt_log_name "diakg2-ib.txt"
```

## Evaluation
Run `eva.py` to evaluate the Relation Extraction (RE) task, and run `eva_ner.py` to evaluate the Named Entity Recognition (NER) task.
