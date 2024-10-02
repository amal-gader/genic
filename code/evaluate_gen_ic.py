import argparse
import os
import sys 
from dotenv import load_dotenv




load_dotenv()
path = os.getenv('VENV_PATH')
sys.path.append(path)


import functools
import logging
import torch
from datasets import Dataset, DatasetDict
from multi_label_classification import stratify_data, quantization_config, tokenize_examples
import numpy as np
from sklearn.metrics import f1_score,  accuracy_score, precision_score
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration, T5TokenizerFast
from peft import PeftModel
from kedataset import ke_Dataset
from tqdm import tqdm
from data_loaders import load_dataset, load_dataset_processed, postprocess


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels', 'text']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels'])
    d['text'] = d['text'] 
    return d



def property_prediction(df,num_labels, ds, relations, id):
    
    base_model_name = 'mistralai/Mistral-7B-v0.1'
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name,  num_labels=num_labels,  quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, f'models/property_prediction/{id}')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to('cuda')
   
    data_collator=functools.partial(collate_fn, tokenizer=tokenizer)
    
    test_dataloader = DataLoader(ds['test'], batch_size=16, shuffle=False, collate_fn=data_collator)

    model.eval() 
    predictions = []
    ground_truths = []
    texts = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Predictions'):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            
            # Get logits from the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Apply a sigmoid to the logits to get the predicted probabilities
            probs = torch.sigmoid(logits)

            pred_labels = (probs > 0.5).int()
            predictions.append(pred_labels.cpu().numpy())
            ground_truths.append(batch['labels'].cpu().numpy())  
            texts.extend(batch['text'])

    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    texts = np.array(texts)

    # Convert binary predictions to class names
    pred_class_names = []
    ground_truth_class_names = []

    for pred, gt in zip(predictions, ground_truths):
        pred_classes = [relations[i] for i in range(len(pred)) if pred[i] == 1]
        gt_classes = [relations[i] for i in range(len(gt)) if gt[i] == 1]
        
        pred_class_names.append(set(pred_classes))
        ground_truth_class_names.append(set(gt_classes))
        
    score=f1_score(ground_truths, predictions, average='micro') 
    precision = precision_score(ground_truths, predictions, average='micro')
    print(f'f1: {score}, precision: {precision}')
     
    df = pd.DataFrame({
    'predicted_properties': pred_class_names,
    #'ground_truth_classes': ground_truth_class_names,
    'head_description': pd.Series(texts).str.extract(r'description:\s*([^,]+)')[0],
    'head': pd.Series(texts).str.extract(r"head:\s*([^,]+)")[0],
    'head_type': pd.Series(texts).str.extract(r'types:\s*(.*?)(?:,\s*description|$)')[0]
    
}) 
 
    return df 
   



def link_prediction(df, id):
    log_threshold = 0.05
    all_labels = []
    all_predictions = []
    hits_at_10 = 0
    hits_at_5 = 0
    total_samples = 0
    val_batch_count = 0
    
    model_name = 't5-large'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = PeftModel.from_pretrained(model, f'models/link_prediction/{id}')
    model.to('cuda')
    
    test_dataset = ke_Dataset(df, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
        

    def predict(batch):
        seq_input, tail_input = batch
        with torch.no_grad():
            # Use beam search to generate the top 10 predictions
            outputs = model.generate(
                input_ids=seq_input['input_ids'].squeeze(1).to('cuda'),
                attention_mask=seq_input['attention_mask'].squeeze(1).to('cuda'),
                max_new_tokens=32,
                num_beams=10,  
                num_return_sequences=10,
                early_stopping=True
            )
            
            # Decode the predictions
            predictions = tokenizer.batch_decode(outputs.to('cpu'), skip_special_tokens=True)
            
            # Get the actual labels and decode them
            labels_on_cpu = tail_input['input_ids'].squeeze(1).cpu()
            labels = np.where(labels_on_cpu != -100, labels_on_cpu, tokenizer.pad_token_id)
            labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            
            # Group predictions into sets of 10
            grouped_predictions = [predictions[i:i + 10] for i in range(0, len(predictions), 10)]
            
        return grouped_predictions, labels

        
    model.eval()
        
    for batch in tqdm(test_dataloader, desc="Evaluate"):
            with torch.no_grad():
                val_batch_count += 1
                grouped_predictions, labels = predict(batch)
                all_labels.extend(labels)
                
                for pred_group, label in zip(grouped_predictions, labels):
                    # Calculate Hits@5 and Hits@10
                    if label in pred_group[:5]:
                        hits_at_5 += 1
                    if label in pred_group[:10]:
                        hits_at_10 += 1
                    total_samples += 1
                    
                    all_predictions.append(pred_group[0])  # Use top prediction for accuracy
            
            val_progress = val_batch_count / len(test_dataloader)
            if val_progress >= log_threshold:
                accuracy = accuracy_score(all_predictions, all_labels)
                print(
                    f"its@1:{accuracy:.4f}",
                    f"Hits@5: {hits_at_5 / total_samples:.4f}",
                    f"Hits@10: {hits_at_10 / total_samples:.4f}"
                )
                log_threshold += 0.05
                
    predicted_df = pd.DataFrame({'prediction':all_predictions, 'label':all_labels})
                
        
    return predicted_df

def test_genic(args):
    id_lp = args.id_lp
    id_pp = args.id_pp
    
    logging.info(f"Loading dataset: {args.dataset}")
    data = load_dataset_processed(args.dataset, args.with_type, args.with_desc)
    df_train, df_val, df_test= stratify_data(data)
    df_train, df_val, df_test = df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

    x_train, y_train = df_train['text'], df_train.drop(columns=['head', 'head types', 'text', 'relation', 'head_description']).values
    x_val, y_val = df_val['text'], df_val.drop(columns=['head', 'head types', 'text', 'relation', 'head_description']).values
    x_test, y_test = df_test['text'], df_test.drop(columns=['head', 'head types', 'text', 'relation', 'head_description']).values
    

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


    all_relations=split1['relation'].unique()

    logging.info("Starting first step testing: Property Prediction")
    
    model_name = 'mistralai/Mistral-7B-v0.1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    tokenized_ds = tokenized_ds.with_format('torch')

    
    df = property_prediction(df_test, num_labels=y_train.shape[1], ds=tokenized_ds, relations=all_relations, id=id_pp)
    
    logging.info("Postprocessing...")
    df=postprocess(df, args.dataset)
    df=df[df['tail'].isnull()==False].rename(columns={'head_type':'head types'})
    df = df.reset_index(drop=True)
    logging.info("Starting second step testing: Link Prediction")
    predicted_triples=link_prediction(df, id=id_lp)
    
    #predicted_triples.to_csv('predictions_fb15k.csv', index=False)
    
  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, choices=['codex', 'fb15k', 'wn18rr'], help='The dataset to be used: codex, fb15k, or wn18rr.')
    parser.add_argument('--id_lp', type=str, help='Checkpoint id to use for link prediction')
    parser.add_argument('--id_pp', type=str, help='Checkpoint id to use for property prediction')
    parser.add_argument('--with_type', action='store_true')
    parser.add_argument('--with_desc', action='store_true')
    args = parser.parse_args()
    test_genic(args)