from run import train_all_subjects

import argparse
import os


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def main():
    parser = argparse.ArgumentParser(description="Chisco In-Subject Neural Decoding")
    
    parser.add_argument('--data_path', type=str, default='../LinJunyi/DATA/chisco/derivatives/preprocessed_pkl', 
                        help='Root directory containing sub-xx folders')
    parser.add_argument('--save_path', type=str, default='./checkpoints_chisco', 
                        help='Where to save per-subject models and logs')
    parser.add_argument('--num_workers', type=int, default=4)
    
    
    parser.add_argument('--bert_path', type=str, default='./models/bert-base-uncased/')
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    parser.add_argument('--init_temperature', type=float, default=0.07, help='Initial temperature value')
    parser.add_argument('--learnable_temp', action='store_true', help='Whether to train the logit_scale')

    parser.add_argument('--uma_threshold', type=float, default=0.8, help='Threshold tau for UMA')
    parser.add_argument('--pooling', type=str, default='cls', choices=['cls', 'mean'], 
                        help='BERT feature extraction strategy: cls or mean pooling')
    
    parser.add_argument('--f1', type=int, default=8)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--c1', type=int, default=64, help='Temporal kernel size')

    hparams = parser.parse_args()

    train_all_subjects(hparams)

if __name__ == "__main__":
    main()