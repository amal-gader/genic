import argparse
import os
from dotenv import load_dotenv
import sys




load_dotenv()
path = os.getenv('VENV_PATH')
sys.path.append(path)

import datetime

import torch.nn.functional as F
import functools
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from data_loaders import load_dataset_processed

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)






quantization_config = BitsAndBytesConfig(
        load_in_4bit = True, 
        bnb_4bit_quant_type = 'nf4',
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.bfloat16
    )

lora_config_pp = LoraConfig(
        r = 16, 
        lora_alpha = 8,
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout = 0.05,
        bias = 'none',
        task_type = 'SEQ_CLS'
    )


freq_threshold = 5 

def split_train_valid_test(df, stratify_col=None, test_size=0.3, random_state=42):
    df = df.reset_index(drop =True)
    df_train, df_temp = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_col
    )
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=random_state )
    return df_train, df_val, df_test
    

def stratify_data(df):
    head_type_counts = df['head types'].value_counts()

    # Filter out tail head types
    common_head_types = head_type_counts[head_type_counts >= freq_threshold].index
    rare_head_types = head_type_counts[head_type_counts < freq_threshold].index
    
    # Split the dataset into freq and unfreq head types
    df_h = df[df['head types'].isin(common_head_types)]
    df_t = df[df['head types'].isin(rare_head_types)]
    head_types_h = df_h['head types']
    df_train_h, df_val_h, df_test_h=split_train_valid_test(df_h, stratify_col=head_types_h)
    df_train_t, df_val_t, df_test_t=split_train_valid_test(df_t)

    # Concatenate the head and tail splits
    df_train = pd.concat([df_train_h, df_train_t])
    df_val = pd.concat([df_val_h, df_val_t])
    df_test = pd.concat([df_test_h, df_test_t])
    
    return df_train, df_val, df_test




def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'])
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs


# custom batch preprocessor
def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels'])
    return d

def compute_metrics(p):
    predictions, labels = p
    # Convert predictions to binary
    binary_predictions = predictions > 0
    
    f1_micro = f1_score(labels, binary_predictions, average='micro')
    f1_macro = f1_score(labels, binary_predictions, average='macro')
    
    precision_micro = precision_score(labels, binary_predictions, average='micro')
    precision_macro = precision_score(labels, binary_predictions, average='macro')
    
    recall_micro = recall_score(labels, binary_predictions, average='micro')
    recall_macro = recall_score(labels, binary_predictions, average='macro')
   
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro
       
    }
    
    

class CustomTrainer(Trainer):

    def __init__(self, label_weights, **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32), pos_weight=self.label_weights)
        return (loss, outputs) if return_outputs else loss   
    

class Train:
    def __init__(self, model, tokenizer, label_weights, tokenized_ds, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.label_weights = label_weights
        self. dataset = dataset
        self.training_args = TrainingArguments(
            output_dir = f'models/property_prediction/multilabel_classification_{self.dataset}',
            learning_rate = 1e-4,
            per_device_train_batch_size = 16,
            per_device_eval_batch_size = 16,
            num_train_epochs = 5,
            weight_decay = 0.01,
            evaluation_strategy = 'epoch'
        )

        self.trainer = CustomTrainer(
            model = self.model,
            args = self.training_args,
            train_dataset = tokenized_ds['train'],
            eval_dataset = tokenized_ds['val'],
            tokenizer = self.tokenizer,
            data_collator = functools.partial(collate_fn, tokenizer=self.tokenizer),
            compute_metrics = compute_metrics,
            label_weights = torch.tensor(self.label_weights, device=model.device)
        )
        
    def train(self):
        self.trainer.train()
        id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{self.dataset}"
        peft_model_id = f"multilabel_mistral_{id}"
        self.trainer.model.save_pretrained(peft_model_id)
        self.tokenizer.save_pretrained(peft_model_id)



def main(args):
    data = load_dataset_processed(args.dataset, args.with_type, args.with_desc)
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
    
    model_name = 'mistralai/Mistral-7B-v0.1'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    tokenized_ds = tokenized_ds.with_format('torch')
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
     
     
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Property prediction Training')
    parser.add_argument('--dataset', type=str, default=None, choices=['codex', 'fb15k', 'wn18rr'], help='The dataset to be used: codex, fb15k, or wn18rr.')
    parser.add_argument('--with_type', action='store_true')
    parser.add_argument('--with_desc', action='store_true')

    args = parser.parse_args()
    main(args)