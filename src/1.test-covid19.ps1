conda env config vars set CLASSPATH="F:\HCMUS\K30\TextAnalytic\Code\stanford-corenlp-latest\stanford-corenlp-4.3.2\stanford-corenlp-4.3.2.jar"
conda activate mypytorchcuda

#$format_lines_out_pattern = "covid19"
$format_bert_out_path = "../bert_data/covid19test10k/covid19test10k"


$model_path = "../model_checkpoints/covid19_211223"
$batch_size = 2000

$test_result_path = "../logs/test_ext_bert_covid19"
$test_from = "../model_checkpoints/covid19_211223/covid19x100krecord_model_step_50000.pt"
## Extractive Setting
python train.py -task ext -mode test -batch_size 2000 -test_batch_size 500 -bert_data_path $format_bert_out_path -log_file "../logs/val_ext_bert_covid19" -model_path $model_path -test_from $test_from -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path $test_result_path
