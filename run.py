import os
import torch
import json
import numpy as np
from dataset.chisco_dataset import prepare_chisco_bert_loaders 
from baseline.EEGNetForBERT import EEGNet
from trainer import ChiscoTrainer

def run_subject_experiment(hparams, subject_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*40}\nTraining Subject: {subject_id}\n{'='*40}")
    
    data_loader = prepare_chisco_bert_loaders(hparams, subject_id)
    train_loader = data_loader['train']
    test_loader = data_loader['test']
    
    model = EEGNet(
        nChan=125, 
        nTime=1651, 
        embedDim=768, 
        dropoutP=0.25, 
        F1=hparams.f1, 
        D=hparams.d
    )
    
    trainer = ChiscoTrainer(model, hparams, device=device)
    
    best_metrics = None
    best_mus = -1.0
    history = []
    
    sub_save_path = os.path.join(hparams.save_path, subject_id)
    os.makedirs(sub_save_path, exist_ok=True)

    for epoch in range(1, hparams.epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
        test_metrics = trainer.evaluate(test_loader)
        
        log_entry = {'epoch': epoch, 'train_loss': train_loss, **test_metrics}
        history.append(log_entry)
        
        if test_metrics['MUS'] > best_mus:
            best_mus = test_metrics['MUS']
            best_metrics = test_metrics
            trainer.save_checkpoint(epoch, sub_save_path, subject_id) 
            print(f">>> Best Model for {subject_id} updated (MUS: {best_mus:.6f})")
            
    with open(os.path.join(sub_save_path, f"{subject_id}_history.json"), 'w') as f:
        json.dump(history, f, indent=4)
        
    return best_metrics

def train_all_subjects(hparams):
    all_subs = sorted([d for d in os.listdir(hparams.data_path) if d.startswith('sub-')])
    print(f"Found {len(all_subs)} subjects: {all_subs}")
    
    summary_results = {}
    
    for sub_id in all_subs:
        best_metrics = run_subject_experiment(hparams, sub_id)
        summary_results[sub_id] = best_metrics
        
    print(f"\n{'='*40}\nFINAL MULTI-SUBJECT SUMMARY\n{'='*40}")
    final_metrics = ["UMA", "MUS", "SRS"]
    for m in final_metrics:
        values = [res[m] for res in summary_results.values()]
        print(f"{m}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
    with open(os.path.join(hparams.save_path, "multi_subject_report.json"), 'w') as f:
        json.dump(summary_results, f, indent=4)