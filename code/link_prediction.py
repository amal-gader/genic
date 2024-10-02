import argparse
import os
import sys 
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import accuracy_score
import torch

from utils import number_of_trainable_model_parameters
load_dotenv()
path = os.getenv('VENV_PATH')
sys.path.append(path)


import datetime
from tqdm import tqdm
from data_loaders import load_dataset
import pandas as pd 
from sklearn.model_selection import train_test_split

from transformers import T5Tokenizer, T5ForConditionalGeneration

from kedataset import ke_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from rich.console import Console
from rich import box
from rich.table import Table, Column
from peft import LoraConfig, get_peft_model, TaskType


console = Console(record=True)
validation_logger = Table(Column("Epoch", justify="center"),
                          Column("Steps", justify="center"),
                          Column("Loss", justify="center"),
                          Column("Accuracy", justify="center"),
                          title="Validation Status", pad_edge=False, box=box.ASCII)



lora_config = LoraConfig(
    r = 16,
    lora_alpha = 8, 
    target_modules = ['q', 'v'],
    lora_dropout = 0.05, 
    bias = 'none',
    task_type=TaskType.SEQ_2_SEQ_LM
)


class LPTrainer(nn.Module):
    def __init__(self,model, tokenizer,dataset, df_train, df_val, num_epochs=3, batch_size=16, with_type=True, with_desc=True):
        super(LPTrainer, self).__init__()
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        train_dataset = ke_Dataset(df_train, self.tokenizer, with_type=with_type, with_desc=with_desc)
        val_dataset = ke_Dataset(df_val, self.tokenizer, with_type=with_type, with_desc=with_desc)
        self.batch_size = batch_size
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        
    def train(self):
        
        val_batch_count = 0
        self.model.train()
        
        for epoch in range(self.num_epochs):
    
            for batch in tqdm(self.train_dataloader, desc="Training batches"):

                seq_input, tail_input = batch
                
                
                tail_input = tail_input['input_ids'].squeeze(1).to('cuda')
                seq_input = seq_input['input_ids'].squeeze(1).to('cuda')
               
                outputs = self.model(input_ids=seq_input, labels=tail_input)
                t5_loss_value = outputs.loss
                
                
                self.optimizer.zero_grad()
                t5_loss_value.backward()
                self.optimizer.step()
                
            val_batch_count, acc, console = self.validate(epoch, val_batch_count)
            
        id = f"{self.dataset}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
        self.model.save_pretrained(f"models/link_prediction/model_{id}")
        self.tokenizer.save_pretrained(f"models/link_prediction/tokenizer_{id}")
        console.save_text(f"models/link_prediction/logs_{id}.txt")
        
    def validate(self, epoch, val_batch_count):
        log_threshold = 0.05
        epoch_val_labels = []
        epoch_val_predictions = []
        accuracy = 0
        hits_at_10 = 0
        hits_at_5 = 0
        total_samples = 0
        
        self.model.eval()
        
        for batch in tqdm(self.val_dataloader, desc="Validating batches"):
            with torch.no_grad():
                val_batch_count += 1
                grouped_predictions, labels = self.predict(batch)
                epoch_val_labels.extend(labels)
                
                for pred_group, label in zip(grouped_predictions, labels):
                    if label in pred_group[:5]:  # Check if label is in top 5 prediction
                        hits_at_5 += 1
                    if label in pred_group[:10]:  # Check if label is in top 10 predictions
                        hits_at_10 += 1
                    total_samples += 1
                    
                    epoch_val_predictions.append(pred_group[0])  # Use top prediction for accuracy
            
            val_progress = val_batch_count / len(self.val_dataloader)
            if val_progress >= log_threshold:
                accuracy = accuracy_score(epoch_val_predictions, epoch_val_labels)
                validation_logger.add_row(
                    str(epoch + 1),
                    str(val_batch_count),
                    f"its@1:{accuracy:.4f}",
                    f"Hits@5: {hits_at_5 / total_samples:.4f}",
                    f"Hits@10: {hits_at_10 / total_samples:.4f}"
                )
                log_threshold += 0.05
                console.print(validation_logger)
        
        return val_batch_count, accuracy, console


    def predict(self, batch):
        seq_input, tail_input = batch
        with torch.no_grad():
            # Use beam search to generate the top 10 predictions
            outputs = self.model.generate(
                input_ids=seq_input['input_ids'].squeeze(1).to(self.device),
                attention_mask=seq_input['attention_mask'].squeeze(1).to(self.device),
                max_new_tokens=32,
                num_beams=10,  # Beam search with 10 beams
                num_return_sequences=10,  # Return the top 10 sequences
                early_stopping=True
            )
        
            predictions = self.tokenizer.batch_decode(outputs.to('cpu'), skip_special_tokens=True)
            
            # Get the actual labels and decode them
            labels_on_cpu = tail_input['input_ids'].squeeze(1).cpu()
            labels = np.where(labels_on_cpu != -100, labels_on_cpu, self.tokenizer.pad_token_id)
            labels = [self.tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            
            # Group predictions into sets of 10
            grouped_predictions = [predictions[i:i + 10] for i in range(0, len(predictions), 10)]
            
        return grouped_predictions, labels

        


def main(args):
    train_data, val_data, test_data = load_dataset(args.dataset)
    with_type = args.with_type
    with_desc = args.with_desc
    data = pd.concat([train_data, val_data, test_data]).reset_index(drop=True)

    # Split the data
    df_train, df_test = train_test_split(data, random_state=42)
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)

    model_name = 't5-large'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = get_peft_model(model, lora_config)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model.to('cuda')
    print(number_of_trainable_model_parameters(model))
    # Train the model
    LPTrainer(model, tokenizer, args.dataset, train_data, val_data, with_type=with_type, with_desc=with_desc).train()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link prediction Training')
    parser.add_argument('--dataset', type=str, default=None, choices=['codex', 'fb15k', 'wn18rr'], help='The dataset to be used: codex, fb15k, or wn18rr.')
    parser.add_argument('--with_type', action='store_true')
    parser.add_argument('--with_desc', action='store_true')
    args = parser.parse_args()
    main(args)
        
