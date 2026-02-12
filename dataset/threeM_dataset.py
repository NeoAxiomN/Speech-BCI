import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import glob
import os
from sklearn.model_selection import train_test_split

class ThreeMDataset(Dataset):
    def __init__(self, data_list):
        super(ThreeMDataset, self).__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample, label = self.data_list[idx]
        return sample, label

    def collate(self, batch):
        samples = np.array([x[0] for x in batch])
        labels = np.array([x[1] for x in batch])
        
        samples = torch.from_numpy(samples).float() 
        labels = torch.from_numpy(labels).long()
        
        return samples, labels

def prepare_3m_subject_dataset(hparams, subject_id):
    sub_path = os.path.join(hparams.data_path, subject_id)
    pkl_files = glob.glob(os.path.join(sub_path, "*.pkl"))
    
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found for {subject_id} in {sub_path}")

    all_trials = []
    for f in pkl_files:
        with open(f, 'rb') as rb:
            content = pickle.load(rb)
            # data shape: (N, 32, 1000), label shape: (N,)
            data = content['data']
            labels = content['label']
            for i in range(len(labels)):
                all_trials.append((data[i], labels[i]))

    train_list, test_val_list = train_test_split(
        all_trials, test_size=0.2, random_state=42, shuffle=True
    )
    val_list, test_list = train_test_split(
        test_val_list, test_size=0.5, random_state=42, shuffle=True
    )

    train_dataset = ThreeMDataset(train_list)
    val_dataset = ThreeMDataset(val_list)
    test_dataset = ThreeMDataset(test_list)

    loaders = {
        'train': DataLoader(
            train_dataset, 
            batch_size=hparams.batch_size, 
            shuffle=True, 
            num_workers=hparams.num_workers,
            collate_fn=train_dataset.collate
        ),
        'val': DataLoader(
            val_dataset, 
            batch_size=hparams.batch_size, 
            shuffle=False, 
            num_workers=hparams.num_workers,
            collate_fn=val_dataset.collate
        ),
        'test': DataLoader(
            test_dataset, 
            batch_size=hparams.batch_size, 
            shuffle=False, 
            num_workers=hparams.num_workers,
            collate_fn=test_dataset.collate
        )
    }

    return loaders