pip install -r requirements.txt
# Set ENV variables
conda env config vars set CLASSPATH="F:\HCMUS\K30\TextAnalytic\Code\stanford-corenlp-latest\stanford-corenlp-4.3.2\stanford-corenlp-4.3.2.jar"
conda activate base


setx CLASSPATH "F:\HCMUS\K30\TextAnalytic\Code\stanford-corenlp-latest\stanford-corenlp-4.3.2\stanford-corenlp-4.3.2.jar"


# Step 3. Sentence Splitting and Tokenization

python preprocess.py -mode tokenize -raw_path "../raw_stories" -save_path "../merged_stories_tokenized"

# Step 4. Format to Simpler Json Files
python preprocess.py -mode format_to_lines -raw_path "../merged_stories_tokenized" -save_path "../json_data/cnndm" -n_cpus 1 -use_bert_basic_tokenizer false -map_path "../urls"

# Step 5. Format to PyTorch Files
python preprocess.py -mode format_to_bert -raw_path "../json_data" -save_path "../bert_data"  -lower -n_cpus 1 -log_file "../logs/preprocess.log"



# TRAIN
## Extractive Setting
python train.py -task ext -mode train -bert_data_path "../bert_data" -ext_dropout 0.1 -model_path MODEL_PATH -lr 2e-3 -visible_gpus 0,1,2 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_bert_cnndm -use_interval true -warmup_steps 10000 -max_pos 512
