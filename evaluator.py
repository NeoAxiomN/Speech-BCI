import torch
import torch.nn.functional as F
import numpy as np

class ChiscoEvaluator:
    def __init__(self, threshold=0.80, device='cuda'):
        self.threshold = threshold
        self.device = device

    @torch.no_grad()
    def calculate_metrics(self, eeg_embeds, text_embeds):
        eeg_embeds = F.normalize(eeg_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        # cosine_similarity 
        pair_sims = F.cosine_similarity(eeg_embeds, text_embeds)

        # Unit Matching Accuracy (UMA)
        uma = (pair_sims > self.threshold).float().mean()

        # Mean Unit Similarity (成对相似度的平均值, MUS)
        mus = pair_sims.mean()

        # Sentence Reconstruction Similarity (生成的句子与参考句子的相似度, SRS)
        srs = mus 

        return {
            'UMA': uma.item(),
            'MUS': mus.item(),
            'SRS': srs.item()
        }

    def calculate_mus_exp(self, text_embeds):
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        n = text_embeds.size(0)
        if n < 2: return 0.0
        # 计算所有样本间的相似度矩阵 (n, n)
        sim_matrix = torch.matmul(text_embeds, text_embeds.t())
        
        # 去掉对角线（自相关）后取平均
        mask = torch.eye(n).to(self.device).bool()
        off_diag_sims = sim_matrix[~mask]
        return off_diag_sims.mean().item()