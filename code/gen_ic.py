import argparse
import os
import sys 
from dotenv import load_dotenv


load_dotenv()
path = os.getenv('VENV_PATH')
sys.path.append(path)



import functools
import logging
from datasets import Dataset, DatasetDict
from multi_label_classification import stratify_data
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration

from multi_label_classification import tokenize_examples, Train, quantization_config, lora_config_pp
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from data_loaders import load_dataset, load_dataset_processed
from link_prediction import LPTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def train_genic(args):
    
    with_type = args.with_type
    with_desc = args.with_desc 
    
    logging.info(f"Loading dataset: {args.dataset}")
    data = load_dataset_processed(args.dataset, with_type, with_desc)
    
    logging.info("Processing the data.")
    df_train, df_val, df_test= stratify_data(data)
    df_train, df_val, df_test = df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

    x_train, y_train = df_train['text'], df_train.drop(columns=['head', 'head types', 'text', 'relation', 'head_description']).values
    x_val, y_val = df_val['text'], df_val.drop(columns=['head', 'head types', 'text', 'relation', 'head_description']).values
    x_test, y_test = df_test['text'], df_test.drop(columns=['head', 'head types', 'text', 'relation', 'head_description']).values

    label_weights = 1 - y_train.sum(axis=0) / y_train.sum()

    train_dataset = Dataset.from_dict({'text': x_train.tolist(), 'labels': y_train.tolist()})
    val_dataset = Dataset.from_dict({'text': x_val.tolist(), 'labels': y_val.tolist()})
    test_dataset = Dataset.from_dict({'text': x_test.tolist(), 'labels': y_test.tolist()})

    
    ds = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
        })
    
    split1, split2, split3 = load_dataset(args.dataset)

    data = pd.concat([split1, split2, split3]).reset_index(drop=True)

    train_data= data[data['head'].isin(df_train['head'].unique())].reset_index(drop=True)
    val_data= data[data['head'].isin(df_val['head'].unique())].reset_index(drop=True)

    
    # Train PP
    logging.info("Starting first step training: Property Prediction")
    model_name = 'mistralai/Mistral-7B-v0.1'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    tokenized_ds = tokenized_ds.with_format('torch')
    
    logging.info("Loading model for Property Prediction")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=y_train.shape[1],
        quantization_config=quantization_config,
    )
        
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config_pp)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to('cuda')
    trainer = Train(model, tokenizer, label_weights, tokenized_ds, args.dataset)
    trainer.train()



    #### Train LP
    logging.info("Starting second step training: Link Prediction")
    lora_config = LoraConfig(
        r = 16,
        lora_alpha = 8, 
        target_modules = ['q', 'v'],
        lora_dropout = 0.05, 
        bias = 'none',
        task_type=TaskType.SEQ_2_SEQ_LM
    )


    model_name = 't5-large'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = get_peft_model(model, lora_config)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model.to('cuda')
    LPTrainer(model, tokenizer, args.dataset, train_data, val_data, with_type=with_type, with_desc=with_desc).train()
    logging.info("Training complete.")
    


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type= str, default=None, choices=['codex', 'fb15k', 'wn18rr'], help='The training dataset')
    parser.add_argument('--with_type', action='store_true')
    parser.add_argument('--with_desc', action='store_true')
    args = parser.parse_args()
    train_genic(args)


        
    

    

   

