conda env config vars set CLASSPATH="F:\HCMUS\K30\TextAnalytic\Code\stanford-corenlp-latest\stanford-corenlp-4.3.2\stanford-corenlp-4.3.2.jar"
conda activate mypytorchcuda

Write-Host $env:CLASSPATH

$story_path = "../raw_stories/covid19"
$tokenized_out_path = "../merged_stories_tokenized/covid19"

$format_lines_out_path = "../json_data/covid19"
$format_lines_out_pattern = "covid19"
$format_bert_out_path = "../bert_data/covid19"

$train_split_percent = 0.8

# Step 3. Sentence Splitting and Tokenization

python preprocess.py -mode tokenize -raw_path $story_path -save_path $tokenized_out_path -n_cpus 14

# Step 4. Format to Simpler Json Files
python preprocess.py -mode format_to_lines -raw_path $tokenized_out_path -save_path "$format_lines_out_path/$format_lines_out_pattern" -n_cpus 14 -use_bert_basic_tokenizer false -map_path "../urls" -train_split $train_split_percent

# Step 5. Format to PyTorch Files
python preprocess.py -mode format_to_bert -raw_path $format_lines_out_path -save_path $format_bert_out_path  -lower -n_cpus 14 -log_file "../logs/preprocess.log"


