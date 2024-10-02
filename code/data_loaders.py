import pandas as pd
import json


from utils import convert_entities, create_text_col, expand_df_with_tails, extract_type, head_types_from_occupation, shorten, update_type, convert_triple

def postprocess(df, dataset):
    if dataset=='codex':
        head_dict = triplets_dict_codex()
    if dataset=='fb15k':
        head_dict = triplets_dict_fb15k()
    if dataset=='wn18rr':
        head_dict=triplets_dict_wn()
    
    df['predicted_properties'] = df['predicted_properties'].apply(list)
    df_expanded = df.explode('predicted_properties').reset_index(drop=True)
    df_expanded = df_expanded.rename(columns={'predicted_properties': 'relation'})
    resulted_df = expand_df_with_tails(df_expanded, head_dict)
    return resulted_df




def process_fb15k(with_type=True, with_desc=True):
    
    train_data = load_fb15k('train')
    val_data = load_fb15k('valid')
    test_data = load_fb15k('test')
    data = pd.concat([train_data, test_data]).reset_index(drop=True)
    all_relations = data['relation'].unique()
    data = data.drop_duplicates(subset=['head', 'relation'])
    data = data.groupby(['head', 'head_description'])[['relation', 'head types']].agg(set).reset_index()
    
    new_columns = {}
    for relation in all_relations:
        new_columns[relation] = data['relation'].apply(lambda x: 1 if relation in x else 0)  
    new_columns_df = pd.DataFrame(new_columns)
    data = pd.concat([data, new_columns_df], axis=1)
    
    data['head types'] = data.apply(lambda row:', '.join(row['head types']), axis=1 )
    
    data['text'] = data.apply(lambda row: create_text_col(row, with_type, with_desc), axis=1)
    
    return data


def load_fb15k(split: str):
    
    path ='../Datasets/FB15K-237'
    entities_dict = {}
    with open(path+'/entity2text.txt', 'r') as file:
        for line in file:
            eid, text = line.strip().split(None, 1)
            entities_dict[eid]=text
            
    desc_dict = {}
    with open(path+'/entity2textlong.txt', 'r') as file:
        for line in file:
            eid, text = line.strip().split(None, 1)
            desc_dict[eid]=text
            
    with open(f'{path}/{split}.txt', 'r') as file:
        triples = file.readlines()
        
    converted_triples = {}
    for i, triple in enumerate(triples):
        head_id, relation_id, tail_id = triple.strip().split()
        description = desc_dict.get(head_id, None)
        head_mention, tail_mention= convert_entities(head_id, tail_id, entities_dict)
        converted_triples[i] = {'head':head_mention,'head_description': description,'relation': relation_id, 'tail': tail_mention}
        
    df_fb15k = pd.DataFrame.from_dict(converted_triples, orient='index')
    df_fb15k['head types']=df_fb15k.apply(lambda row: extract_type(row['relation']), axis=1)
    df_fb15k['relation'] = df_fb15k['relation'].apply(shorten)
    
    return df_fb15k


def load_codex(split: str):

    with open('../Datasets/codex/entities.json', 'r') as file:
        entities = pd.read_json(file, orient='index')
    entities_dict = {index: [row['label'], row['description']] for index, row in entities.iterrows()}
    with open('../Datasets/codex/relations.json', 'r') as file:
        relations = pd.read_json(file, orient='index')
    relations_dict = {index: row['label'] for index, row in relations.iterrows()}
    with open('../Datasets/codex/types.json', 'r') as file:
        types = pd.read_json(file, orient='index')
    types_dict = {index: row['label'] for index, row in types.iterrows()}
    with open('../Datasets/codex/entity2types.json', 'r') as file:
        ent2types = json.load(file)
    with open(f'../Datasets/codex-l/{split}.txt', 'r') as file:
        triples = file.readlines()
        
    converted_triples = {}
    for i, triple in enumerate(triples):
        head_types_mentions=[]
        head_id, relation_id, tail_id = triple.strip().split()
        head_types = ent2types[head_id]
        for type in head_types:
            head_types_mentions.append(types_dict[type])
        head_mention, relation_mention, tail_mention, head_description = convert_triple(head_id, relation_id, tail_id, relations_dict, entities_dict)
        converted_triples[i] = {'head':head_mention, 'relation': relation_mention, 'tail': tail_mention, 'head types': ', '.join(head_types_mentions),'head_description': head_description}
        
    df_codex = pd.DataFrame.from_dict(converted_triples, orient='index')
    head_types = head_types_from_occupation(df_codex)
    df_codex['head types'] = df_codex.apply(update_type,head_types=head_types, axis=1)
    
    return df_codex

def triples_dict(path, entities_dict, relations_dict):
    
    head_to_triplets = {}
    for split in ['train', 'valid', 'test']:
        with open(f'{path}/{split}.txt', 'r') as file:
            for line in file:
                data = file.readlines()
        for line in data:
            head, relation, tail = line.strip().split()
            if head not in head_to_triplets:
                head_to_triplets[head] = []
            head_to_triplets[head].append((relation, tail))
            
    triplets_str = {}
    
    for head, triplets in head_to_triplets.items():
        head_str = entities_dict.get(head, head)  # Convert head to string
        triplets_str[head_str] = []  # Initialize the list for string triplets
        
        for relation, tail in triplets:
            if relations_dict==None:
                relation_str = shorten(relation)
            else:
                relation_str = relations_dict.get(relation, relation)  # Convert relation to string
            tail_str = entities_dict.get(tail, tail)  # Convert tail to string
            triplet_str = (relation_str, tail_str)  # Format as a string triplet
            triplets_str[head_str].append(triplet_str)
            
    return triplets_str
   
def triplets_dict_fb15k():
    path ='../Datasets/FB15K-237'
    entities_dict = {}
    with open(path+'/entity2text.txt', 'r') as file:
        for line in file:
            eid, text = line.strip().split(None, 1)
            entities_dict[eid]=text
    triplets_str = triples_dict(path, entities_dict, relations_dict=None)  
    return triplets_str       
     
   
 
def triplets_dict_wn():
    path = '../Datasets/WN18RR'
    entities_dict = {}
    with open(path+'/entity2text.txt', 'r') as file:
        for line in file:
            eid, text = line.strip().split(None, 1)
            head_mention, head_desc = text.strip().split(',', 1)
            entities_dict[eid]=head_mention
            
    relations_dict={}
    with open(path+'/relation2text.txt', 'r') as file:
        for line in file:
            rid, text = line.strip().split(None, 1)
            relations_dict[rid]=text
        triplets_str = triples_dict(path, entities_dict, relations_dict)  
    return triplets_str

   

def triplets_dict_codex():
    path = '../Datasets/codex-l'
    with open('../Datasets/codex/entities.json', 'r') as file:
        entities = pd.read_json(file, orient='index')
    entities_dict = {index: row['label'] for index, row in entities.iterrows()}
    with open('../Datasets/codex/relations.json', 'r') as file:
        relations = pd.read_json(file, orient='index')
    relations_dict = {index: row['label'] for index, row in relations.iterrows()}
    triplets_str = triples_dict(path, entities_dict, relations_dict)  
    return triplets_str



#TODO clean   
def process_codex(with_type=True, with_desc=True):
    train_data = load_codex('train')
    val_data = load_codex('valid')
    test_data = load_codex('test')
    
    data = pd.concat([train_data, val_data, test_data]).reset_index(drop=True)
    data = data.drop_duplicates(subset=['head', 'relation'])
    data = data.groupby(['head', 'head types', 'head_description'])[['relation']].agg(set).reset_index()
    all_relations = train_data['relation'].unique()
        
    for relation in all_relations:
        data[relation] = data['relation'].apply(lambda x: 1 if relation in x else 0)
            
    data['text'] = data.apply(lambda row: create_text_col(row, with_type, with_desc), axis=1) 
       
    return data


def load_wn18rr(split: str):
    
    path ='../Datasets/WN18RR'
    
    def entity2type():
        with open(f'{path}/wordnet-mlj12-definitions.txt', 'r') as file:
            entity2type = dict()
            for line in file:
                eid, etype, _ = line.strip().split('\t')
                entity2type[eid] = etype[-4:]
        return entity2type
    
    def convert_triple(head_id, relation_id, tail_id, rel_dict, ent_dict):
        head_mention = ent_dict.get(head_id)
        relation_mention = rel_dict.get(relation_id)
        tail_mention = ent_dict.get(tail_id)
        return head_mention, relation_mention, tail_mention
    
    entities_dict = {}
    with open(path+'/entity2text.txt', 'r') as file:
        for line in file:
            eid, text = line.strip().split(None, 1)
            entities_dict[eid]=text
            
    relations_dict={}
    with open(path+'/relation2text.txt', 'r') as file:
        for line in file:
            rid, text = line.strip().split(None, 1)
            relations_dict[rid]=text
   
    with open(f'{path}/{split}.txt', 'r') as file:
        triples = file.readlines()
        
    entity2type = entity2type()  
     
    converted_triples = {}
    for i, triple in enumerate(triples):
        head_id, relation_id, tail_id = triple.strip().split()
        head_type = entity2type[head_id]
        head_mention, relation_mention, tail_mention= convert_triple(head_id, relation_id, tail_id, relations_dict, entities_dict)
        head_mention, head_desc = head_mention.strip().split(',', 1)
        tail_mention, _ = tail_mention.strip().split(',', 1)
        converted_triples[i] = {'head':head_mention,'head_description': head_desc, 'head types': head_type, 'relation': relation_mention, 'tail': tail_mention}
        
    df_wn18rr = pd.DataFrame.from_dict(converted_triples, orient='index')
    
    return df_wn18rr



def process_wn18rr(with_type=True, with_desc=True):
    
    train_data = load_wn18rr('train')
    valid_data = load_wn18rr('valid')
    test_data = load_wn18rr('test')
    data = pd.concat([train_data,valid_data,test_data]).reset_index(drop=True)
    
    all_relations = data['relation'].unique()
    data = data.drop_duplicates(subset=['head', 'relation'])
    data = data.groupby(['head', 'head_description'])[['relation', 'head types']].agg(set).reset_index()
    new_columns = {}

    for relation in all_relations:
        new_columns[relation] = data['relation'].apply(lambda x: 1 if relation in x else 0)
        
    new_columns_df = pd.DataFrame(new_columns)
    data = pd.concat([data, new_columns_df], axis=1)
    data['head types'] = data.apply(lambda row:', '.join(row['head types']), axis=1 )
    data['text'] = data.apply(lambda row: create_text_col(row, with_type, with_desc), axis=1)
    
    return data



def entity2id_fb15k():
    path ='../Datasets/FB15K-237'
    entities_dict = {}
    with open(path+'/entity2text.txt', 'r') as file:
        for line in file:
            eid, text = line.strip().split(None, 1)
            entities_dict[text]=eid
    return entities_dict



def entity2id_wn():

    path ='../Datasets/WN18RR' 
    entities_dict = {}
    with open(path+'/entity2text.txt', 'r') as file:
        for line in file:
            eid, text = line.strip().split(None, 1)
            head, _ = text.strip().split(',', 1)
            entities_dict[head]=eid
                
    relations_dict={}
    with open(path+'/relation2text.txt', 'r') as file:
        for line in file:
            rid, text = line.strip().split(None, 1)
            relations_dict[text]=rid
            
    return entities_dict, relations_dict


def entity2id_codex():
    with open('../Datasets/codex/entities.json', 'r') as file:
            entities = pd.read_json(file, orient='index')
    entities_dict = {row['label']: index for index, row in entities.iterrows()}
    with open('../Datasets/codex/relations.json', 'r') as file:
            relations = pd.read_json(file, orient='index')
    relations_dict = {row['label']: index for index, row in relations.iterrows()}
    return entities_dict, relations_dict

def convertent2id(ent, ent_dict):
    entid = ent_dict.get(ent)
    return entid
    
def convertrel2id(rel, rel_dict):
    relid = rel_dict.get(rel)
    return relid
    
    
    
def load_dataset_processed(dataset_name, with_type=True, with_desc=True):
    
    if dataset_name == 'codex':
        data = process_codex(with_type, with_desc)
        
    elif dataset_name == 'fb15k':
        data = process_fb15k(with_type, with_desc)
      
    elif dataset_name == 'wn18rr':
        data = process_wn18rr(with_type, with_desc)
       
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data



def load_dataset(dataset_name):
    
    if dataset_name == 'codex':
        train_data = load_codex('train')
        val_data = load_codex('valid')
        test_data = load_codex('test')
        
    elif dataset_name == 'fb15k':
        train_data = load_fb15k('train')
        val_data = load_fb15k('valid')
        test_data = load_fb15k('test')
        
    elif dataset_name == 'wn18rr':
        train_data = load_wn18rr('train')
        val_data = load_wn18rr('valid')
        test_data = load_wn18rr('test')
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_data, val_data, test_data
