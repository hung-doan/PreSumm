$format_lines_out_pattern = "covid19"
$format_bert_out_path = "../bert_data/covid19"


$model_path = "../model_checkpoints/covid19"
$batch_size = 2000
$train_steps = 50000

## Extractive Setting
python train.py -task ext -mode train -bert_data_path "$format_bert_out_path/$format_lines_out_pattern" -ext_dropout 0.1 -model_path $model_path -lr 2e-3 -visible_gpus 0 -report_every 50 -save_checkpoint_steps 1000 -batch_size $batch_size -train_steps $train_steps -accum_count 2 -log_file "../logs/ext_bert_$format_lines_out_pattern" -use_interval true -warmup_steps 10000 -max_pos 512