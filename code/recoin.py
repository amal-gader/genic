import argparse
import pandas as pd 
from data_loaders import load_dataset_processed
from sklearn.model_selection import train_test_split
from collections import defaultdict

freq_threshold = 5  


def recoin_data(df):
    head_type_counts = df['head types'].value_counts()
    # Filter out rare head types
    common_head_types = head_type_counts[head_type_counts >= freq_threshold].index
    rare_head_types = head_type_counts[head_type_counts < freq_threshold].index
    # Split the dataset into common and rare head types
    df_common = df[df['head types'].isin(common_head_types)]
    df_rare = df[df['head types'].isin(rare_head_types)]
    head_types_common = df_common['head types'].values
    
    # Stratify on the common head types
    df_train_common, df_test_common = train_test_split(df_common, test_size=0.2, random_state=42, stratify=head_types_common)
    df_train_rare, df_test_rare = train_test_split(df_rare, test_size=0.2, random_state=42)

    df_train = pd.concat([df_train_common, df_train_rare])
    df_test = pd.concat([df_test_common, df_test_rare])
    
    return df_train, df_test


def property_counts(df):

    property_occurrences = defaultdict(lambda: defaultdict(int))
    class_counts = defaultdict(int)

    for index, row in df.iterrows():
        head_types = row['head types'].split(',')
        relations = row['relation']
        
        for head_type in head_types:
            class_counts[head_type] += 1
            for relation in relations:
                property_occurrences[head_type][relation] += 1

    property_occurrences = {k: dict(v) for k, v in property_occurrences.items()}
    class_counts = dict(class_counts)
    return property_occurrences, class_counts


def metrics(ground_truth, predictions):
    predicted_set = set(predictions)
    correct_predictions = ground_truth.intersection(predicted_set)
    precision = len(correct_predictions) / len(predicted_set) if predicted_set else 0.0
    recall = len(correct_predictions) / len(ground_truth) if ground_truth else 0.0
    f1_score = 2 * precision * recall/( precision + recall) if precision or recall else 0.0
    return recall, precision ,f1_score

def predict(row,class_counts, property_counts, threshold=0.4):
    combined_dict={}
    count=0
    missing_types=[]
    properties = []
    for t in row['head types'].split(','):
        if t in class_counts:
            count+=class_counts[t]
            for key, value in property_counts[t].items():
                if key in combined_dict:
                    combined_dict[key] += value  
                else:
                    combined_dict[key] = value            
            for key in combined_dict:
                combined_dict[key] /= count
                if combined_dict[key] > threshold:
                    properties.append(key)
        else:
            missing_types.append(row['head types'])
    return properties


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, choices=['codex', 'fb15k', 'wn18rr'], help='The dataset to be used: codex, fb15k, or wn18rr.')
    args = parser.parse_args()
    
    df = load_dataset_processed(args.dataset)
    
    df_train, df_test = recoin_data(df)
    property_occ, class_counts= property_counts(df_train)
    
    df_test['predicted_properties'] = df_test.apply(lambda row: predict(row,class_counts, property_occ), axis=1)
    df_test[['recall', 'precision', 'f1_score']] = df_test.apply(
        lambda row: pd.Series(metrics(row['relation'], row['predicted_properties'])),
        axis=1
    )
    
    final_recall = df_test['recall'].mean()
    final_precision = df_test['precision'].mean()
    final_f1_score = df_test['f1_score'].mean()
    
    print(f"Final Recall: {final_recall}")
    print(f"Final Precision: {final_precision}")
    print(f"Final F1 Score: {final_f1_score}")
