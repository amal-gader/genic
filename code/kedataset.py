from torch.utils.data import Dataset
 
class ke_Dataset(Dataset):
      def __init__(self, df, tokenizer, corrupt_head=True, with_type=True, with_desc=True):
        self.df=df
        self.head = df['head']
        self.relation = df['relation']
        self.tail = df['tail']
        self.desc = df['head_description']
        self.type = df['head types']
        self.tokenizer = tokenizer
        self.corrupt_head = corrupt_head  
        self.with_type=with_type
        self.with_desc=with_desc

      def __len__(self):
         return len(self.df)
    
      def __getitem__(self, idx):
            head = self.head[idx]
            relation = self.relation[idx]
            tail = self.tail[idx]
            description = self.desc[idx]
            types = self.type[idx]
            
            seq_text = f"head: {head}"
            if self.with_type:
                seq_text += f", types: {types}"
            if self.with_desc:
                seq_text += f", description: {description}"
            seq_text+=f", relation:{relation}, tail:.."
            
           
            tail_input = self.tokenizer(tail, return_tensors='pt',padding='max_length', truncation=True, max_length=32)
            seq_input=self.tokenizer(seq_text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
            return seq_input, tail_input
        
    
