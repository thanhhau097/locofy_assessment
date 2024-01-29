## Installation
```
apt install graphviz

pip install -r requirements.txt
```

## Training
```
python train.py --do_train --do_eval --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 1e-4 --warmup_ratio 0.01 --lr_scheduler_type cosine --save_strategy epoch --evaluation_strategy epoch --logging_strategy steps --logging_steps 20 --save_total_limit 2 --load_best_model_at_end True --optim adamw_torch --weight_decay 1e-2 --num_train_epochs 100 --metric_for_best_model eval_f1 --greater_is_better=True --dataloader_num_workers=8 --max_grad_norm=1.0 --overwrite_output_dir=True --output_dir ./outputs/ --report_to none
```

## Experiment
Baseline result:

| Model | F1 | Precision | Recall |
| --- | --- | --- | --- |
| GCN | 0.70 | 0.83 | 0.61 |