import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from tqdm import tqdm
import numpy as np
import os
from evaluator import ChiscoEvaluator

class ChiscoTrainer:
    def __init__(self, eeg_model, hparams, device='cuda'):
        self.hparams = hparams
        self.device = device
        
        self.eeg_model = eeg_model.to(device)
        
        self.text_model = BertModel.from_pretrained(hparams.bert_path).to(device)
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        # 对比学习的温度系数
        self.logit_scale = nn.Parameter(torch.ones([])*np.log(1/hparams.init_temperature), requires_grad=hparams.learnable_temp)
        self.logit_scale.data = self.logit_scale.data.to(device)
        
        # 指标 
        self.evaluator = ChiscoEvaluator(threshold=hparams.uma_threshold, device=device)
        
        optim_params = [{'params': self.eeg_model.parameters()}]
        if hparams.learnable_temp:
            optim_params.append({'params': [self.logit_scale]})
            print(f"-> Logit scale is LEARNABLE (Initial Temp: {hparams.init_temperature})")
        else:
            print(f"-> Logit scale is FIXED (Value: {self.logit_scale.exp().item():.2f})")
        
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=hparams.lr, 
            weight_decay=hparams.weight_decay
        )

    # def contrastive_loss(self, eeg_features, text_features):
    #     """CLIP-style symmetric contrastive loss"""
    #     eeg_features = F.normalize(eeg_features, p=2, dim=-1)
    #     text_features = F.normalize(text_features, p=2, dim=-1)

    #     n = eeg_features.shape[0]
    #     # 计算相似度矩阵并缩放
    #     logits = torch.matmul(eeg_features, text_features.t()) * self.logit_scale.exp()
    #     labels = torch.arange(n).to(self.device)
        
    #     loss_eeg = F.cross_entropy(logits, labels)
    #     loss_text = F.cross_entropy(logits.t(), labels)
    #     return (loss_eeg + loss_text) / 2
    
    def contrastive_loss(self, eeg_features, text_features):
        eeg_norm = F.normalize(eeg_features, p=2, dim=-1)
        text_norm = F.normalize(text_features, p=2, dim=-1)

        # 计算目标分布
        with torch.no_grad():
            t2t_sim = torch.matmul(text_norm, text_norm.t()) * self.logit_scale.exp()
            targets = F.softmax(t2t_sim, dim=-1)

        # 计算预测分布
        # logits 是 EEG(行) 与 Text(列) 的匹配度
        logits_eeg = torch.matmul(eeg_norm, text_norm.t()) * self.logit_scale.exp()
        # logits_text 是 Text(行) 与 EEG(列) 的匹配度 (转置)
        logits_text = logits_eeg.t()
        
        # 双向 KL 散度
        loss_eeg = -torch.sum(targets * F.log_softmax(logits_eeg, dim=-1), dim=-1).mean()
        loss_text = -torch.sum(targets * F.log_softmax(logits_text, dim=-1), dim=-1).mean()
        
        return (loss_eeg + loss_text) / 2

    def _get_text_features(self, input_ids, mask):
        bert_output = self.text_model(input_ids=input_ids, attention_mask=mask)
        
        if self.hparams.pooling == 'cls':
            # 取 [CLS] token (B, 768)
            return bert_output.last_hidden_state[:, 0, :]
        else: # self.hparams.pooling == 'mean':
            # (B, L, 768) -> (B, 768)
            embeddings = bert_output.last_hidden_state
            mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask


    def train_epoch(self, train_loader, epoch):
        self.eeg_model.train()
        total_loss = 0
        
        # 显式打印当前 Epoch
        print(f"\nEpoch [{epoch}/{self.hparams.epochs}]")
        
        # leave=False 会在进度条完成后自动消去
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in pbar:
            eeg = batch['eeg'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # EEG 特征提取 
            eeg_features = self.eeg_model(eeg) 
            
            # 获取 BERT 语义嵌入
            with torch.no_grad():
                text_features = self._get_text_features(input_ids=input_ids, mask=mask)
            
            loss = self.contrastive_loss(eeg_features, text_features)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.logit_scale.clamp_(0, np.log(100))
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f} | Scale: {self.logit_scale.exp().item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"Train Summary => Loss: {avg_loss:.6f} | Scale: {self.logit_scale.exp().item():.4f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.eeg_model.eval()
        epoch_metrics = {'UMA': [], 'MUS': [], 'SRS': []}
        mus_exp_list = []
        
        # 验证进度条也设置为完成即消去
        pbar_eval = tqdm(test_loader, desc="Evaluating", leave=False)
        
        for batch in pbar_eval:
            eeg = batch['eeg'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            
            eeg_feat = self.eeg_model(eeg)
            text_feat = self._get_text_features(input_ids=input_ids, mask=mask)
            
            res = self.evaluator.calculate_metrics(eeg_feat, text_feat)
            m_exp = self.evaluator.calculate_mus_exp(text_feat)
            for k in epoch_metrics.keys():
                epoch_metrics[k].append(res[k])
            mus_exp_list.append(m_exp)
                
        final_results = {k: np.mean(v) for k, v in epoch_metrics.items()}
        final_results['MUS_exp'] = np.mean(mus_exp_list)
        print(f"Test Metrics  => UMA: {final_results['UMA']:.6f} | MUS: {final_results['MUS']:.6f} "
              f"| SRS: {final_results['SRS']:.6f} | MUS_exp: {final_results['MUS_exp']:.6f}")
        return final_results

    def save_checkpoint(self, epoch, path, sub_id):
        state = {
            'epoch': epoch,
            'subject': sub_id,
            'eeg_model_state_dict': self.eeg_model.state_dict(),
            'logit_scale': self.logit_scale.data,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        save_file = os.path.join(path, f"best_model_{sub_id}.pt")
        torch.save(state, save_file)