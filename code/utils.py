from collections import defaultdict
from dotenv import load_dotenv
import logging
import pandas as pd



logging.basicConfig(level=logging.INFO)

load_dotenv()

def convert_triple(head_id, relation_id, tail_id, rel_dict, ent_dict):
    head_mention = ent_dict.get(head_id)[0]
    head_description = ent_dict.get(head_id)[1]
    relation_mention = rel_dict.get(relation_id)
    tail_mention = ent_dict.get(tail_id)[0]
    return head_mention, relation_mention, tail_mention, head_description

def number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def extract_type(relation: str):
    substrings = relation.split('/')
    return '/'.join(substrings[:3])

def shorten(relation: str):
    substrings = relation.split('/')
    return '/'.join(substrings[3:])

def convert_entities(head_id, tail_id, ent_dict):
    head_mention = ent_dict.get(head_id)
    tail_mention = ent_dict.get(tail_id)
    return head_mention, tail_mention

def add_occupations(row, entity_dict):
    head = row['head']
    head_types = row['head types']
    relations = row['relation']
    if 'human' in head_types and 'occupation' in relations:
            occupations = [tail for relation, tail in entity_dict.get(head) if relation == 'occupation']
            occupations=', '.join(occupations) 
            head_types=head_types+', '+occupations
    return head_types


def create_text_col(row, with_type, with_desc):
    text = f"head: {row['head']}"
    if with_type:
        text += f", types: {row['head types']}"
    if with_desc:
        text += f", description: {row['head_description']}"
    return text

def head_types_from_occupation(df):
    head_types = defaultdict(set) 
    for _, row in df.iterrows():
        head = row['head']
        if row['head types'] != 'NA':
            head_types[head].add(row['head types'])
        if row['relation'] == 'occupation' and row['tail'] != 'NA':
            head_types[head].add(row['tail'])
    head_types = {k: list(v) for k, v in head_types.items()}
    return head_types

def update_type(row, head_types):
    head = row['head']
    if head in head_types:
        return ', '.join(head_types[head])
    else:
        return row['head types'] 
    
    
def get_tails_for_row(row, head_dict):
    head = row['head']
    relation = row['relation']
    tails = []
    if head in head_dict:
        for rel, tail in head_dict[head]:
            if rel == relation:
                tails.append(tail)
    if not tails:
        tails.append(None)
    return tails
    
def expand_df_with_tails(df, head_dict):
    new_rows = []
    for _, row in df.iterrows():
        tails = get_tails_for_row(row, head_dict)
        for tail in tails:
            new_row = {
                'head': row['head'],
                'relation': row['relation'],
                'head_type': row['head_type'],
                'head_description': row['head_description'],
                'tail': tail
            }
            new_rows.append(new_row)
    df_result = pd.DataFrame(new_rows)
    return df_result


