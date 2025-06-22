import copy
import pickle
import torch
import numpy as np
import pandas as pd
import os
import json
import pickle
import torch.nn.functional as F
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score
import itertools
from find_critical_parameters import find_critical_para, load_model

def cos_sim_toxic(model_id, df ,gradient_norms_compare, minus_row, minus_col):

    columns = df[['user_input', 'toxicity']]
    model, tokenizer = load_model(model_id)
    
    #  Prompt templates
    sep_token, sep_token_id = tokenizer.unk_token, tokenizer.unk_token_id
    prompt = (
        f'<s>[INST] <<SYS>> {{system_prompt}} <</SYS>> {{content}} [/INST]' + 
        f'{{sep_token}} {{summary}} {{eos_token}}'
    )
    def apply_prompt_template(sample):
        txt = prompt.format(
            system_prompt='You are a helpful assistant. Help me with the following query: ',
            content=sample['source'],
            summary=sample['target'],
            eos_token=tokenizer.eos_token,
            sep_token=sep_token,
        )
        return {
            'text': txt,
        }     

    #  Calculate cosine similarity based score 
    cos_all = []
    label_all = []

    print("[INFO] 코사인 유사도 계산 시작. 총 샘플 수:", len(columns))
    for index, row in tqdm(columns.iterrows(), total=len(columns), desc="샘플 처리 중"):
        torch.cuda.empty_cache()
        # Ground truth label
        if row['toxicity'] == 1:
            label_all.append(1)
        else: 
            label_all.append(0)


        # Gradient for prompt paired with complaince response Sure
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        basic_sample = {}
        basic_sample["source"] = row['user_input']
        basic_sample["target"] = "Sure"
        d = apply_prompt_template(basic_sample)
        input_ids = tokenizer(d['text']).input_ids
        sep = input_ids.index(sep_token_id)
        
        input_ids = input_ids[:sep] + input_ids[sep+1:]
        input_ids = torch.tensor(np.array([input_ids]))
        input_ids = input_ids.to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :sep] = -100
        target_ids = target_ids.to(model.device)
        optimizer.zero_grad()
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
        neg_log_likelihood.backward()
        
        # Cosine similarities for safety-critical parameters
        cos = []
        import torch.nn.functional as F
        for name, param in model.named_parameters():
            if  param.grad is not None and ("mlp" in name or "self" in name):
                grad_norm = param.grad.to(gradient_norms_compare[name].device)
                row_cos = torch.nan_to_num(F.cosine_similarity(grad_norm, (gradient_norms_compare[name]), dim=1))
                col_cos = torch.nan_to_num(F.cosine_similarity(grad_norm, (gradient_norms_compare[name]), dim=0))
                ref_row = minus_row[name]
                ref_col = minus_col[name]
                cos.extend(row_cos[ref_row>1].cpu().tolist())
                cos.extend(col_cos[ref_col>1].cpu().tolist())

        cos_all.append(cos)
        
    # For GradSafe-Zero, average all cosine similarites as one score
    cos_all = [sum(feature)/len(feature) for feature in cos_all]
    
    # Calculate AUPRC
    precision, recall, thresholds = precision_recall_curve(label_all, cos_all)
    auprc = auc(recall, precision)

    # Threshold 최적화
    print("[INFO] Threshold 최적화 시작...")
    thresholds_test = np.linspace(0.05, 0.5, 50)
    f1_scores = []
    precisions = []
    recalls = []
    
    for thresh in tqdm(thresholds_test, desc="Threshold 테스트 중"):
        pred = [1 if score >= thresh else 0 for score in cos_all]
        f1_scores.append(f1_score(label_all, pred, zero_division=0))
        precisions.append(precision_score(label_all, pred, zero_division=0))
        recalls.append(recall_score(label_all, pred, zero_division=0))
    
    # 최적 threshold 찾기
    best_threshold = thresholds_test[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    
    print(f"[INFO] 최적 threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
    
    # Threshold 분석 시각화
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Precision-Recall Curve
    plt.subplot(2, 2, 1)
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.fill_between(recall, precision, alpha=0.3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUPRC: {auprc:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Score Distribution
    plt.subplot(2, 2, 2)
    toxic_scores = [cos_all[i] for i in range(len(label_all)) if label_all[i] == 1]
    non_toxic_scores = [cos_all[i] for i in range(len(label_all)) if label_all[i] == 0]
    
    plt.violinplot([toxic_scores], positions=[1], showmeans=True, showextrema=True)
    plt.violinplot([non_toxic_scores], positions=[2], showmeans=True, showextrema=True)
    plt.xticks([1, 2], ['Toxic', 'Non-toxic'])
    plt.ylabel('Cosine Similarity Score')
    plt.title('Score Distribution by Query Type')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Threshold Analysis
    plt.subplot(2, 2, 3)
    plt.plot(thresholds_test, f1_scores, 'g-', label='F1 Score', linewidth=2)
    plt.plot(thresholds_test, precisions, 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds_test, recalls, 'r-', label='Recall', linewidth=2)
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Score vs Threshold scatter
    plt.subplot(2, 2, 4)
    plt.scatter([cos_all[i] for i in range(len(label_all)) if label_all[i] == 0], 
                [0] * sum(1 for x in label_all if x == 0), 
                alpha=0.6, label='Non-toxic', s=20)
    plt.scatter([cos_all[i] for i in range(len(label_all)) if label_all[i] == 1], 
                [1] * sum(1 for x in label_all if x == 1), 
                alpha=0.6, label='Toxic', s=20)
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
    plt.xlabel('Cosine Similarity Score')
    plt.ylabel('True Label')
    plt.title('Score Distribution with Optimal Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'toxic_chat_threshold_analysis_{model_id.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 최적 threshold로 최종 성능 계산
    predicted_labels = [1 if score >= best_threshold else 0 for score in cos_all]
    precision = precision_score(label_all, predicted_labels)
    recall = recall_score(label_all, predicted_labels)
    f1 = f1_score(label_all, predicted_labels)
    
    print("=== 최적 Threshold 적용 결과 ===")
    print(f"최적 threshold: {best_threshold:.3f}")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUPRC:", auprc)
    
    del gradient_norms_compare
    del cos_all
    del label_all
    return auprc, f1


if __name__ == "__main__":
    for model_id in ['./model/Llama-2-7b-chat-hf']:
        gradient_norms_compare, minus_row_cos, minus_col_cos =  find_critical_para(model_id)
        df = pd.read_csv('./data/toxic-chat/toxic-chat_annotation_test.csv')
        auprc, f1 = cos_sim_toxic(model_id, df,gradient_norms_compare, minus_row_cos, minus_col_cos)