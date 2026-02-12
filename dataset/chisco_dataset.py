import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import glob
import os
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

class ChiscoBertDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=32):
        super(ChiscoBertDataset, self).__init__()
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_dict = self.data_list[idx]
        
        # EEG: (1, 125, 1651)
        x = sample_dict['input_features']
        
        text = sample_dict['text']
        encoded_text = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'eeg': torch.from_numpy(x).float(),
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'raw_text': text
        }

def prepare_chisco_bert_loaders(hparams, subject_id):
    tokenizer = BertTokenizer.from_pretrained(hparams.bert_path)

    sub_path = os.path.join(hparams.data_path, subject_id, 'eeg')
    pkl_files = glob.glob(os.path.join(sub_path, "*task-imagine*.pkl"))
    
    all_trials = []
    for f in pkl_files:
        with open(f, 'rb') as rb:
            all_trials.extend(pickle.load(rb))

    train_list, test_list = train_test_split(all_trials, test_size=0.2, random_state=42)

    datasets = {
        'train': ChiscoBertDataset(train_list, tokenizer),
        'test': ChiscoBertDataset(test_list, tokenizer)
    }

    loaders = {
        k: DataLoader(
            v, 
            batch_size=hparams.batch_size, 
            shuffle=(k == 'train'),
            num_workers=hparams.num_workers
        ) for k, v in datasets.items()
    }

    return loaders