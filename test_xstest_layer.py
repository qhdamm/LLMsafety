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

def cos_sim_xstest(model_id, df, gradient_norms_compare, minus_row, minus_col):
    columns = df[['prompt', 'type']]
    model, tokenizer = load_model(model_id)

    # Prompt templates
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

    # Calculate cosine similarity based score with layer-wise differences
    cos_all = []
    layer_diff_all = []
    label_all = []
    layer_wise_stats = {0: {}, 1: {}}  
   
    for index, row in tqdm(columns.iterrows(), total=len(columns), desc="Processing prompts"):
        # Ground truth label
        if "contrast" in row['type']:
            label = 1
            label_all.append(1)
        else: 
            label = 0
            label_all.append(0)
        
        # Gradient for prompt paired with compliance response "Sure"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        basic_sample = {}
        basic_sample["source"] = row["prompt"]
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
        
        # Group parameters by layer
        layer_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None and ("mlp" in name or "self" in name):
                # Extract layer number from parameter name
                layer_num = int(name.split('.')[2]) if 'layers' in name else 0
                
                if layer_num not in layer_grads:
                    layer_grads[layer_num] = []
                
                grad_norm = param.grad.to(gradient_norms_compare[name].device)
                row_cos = torch.nan_to_num(F.cosine_similarity(grad_norm, gradient_norms_compare[name], dim=1))
                col_cos = torch.nan_to_num(F.cosine_similarity(grad_norm, gradient_norms_compare[name], dim=0))
                ref_row = minus_row[name]
                ref_col = minus_col[name]
                
                # Store cosine similarities
                cos.extend(row_cos[ref_row>1].cpu().tolist())
                cos.extend(col_cos[ref_col>1].cpu().tolist())
                
                # Store layer-wise information
                layer_grads[layer_num].extend(row_cos[ref_row>1].cpu().tolist())
                layer_grads[layer_num].extend(col_cos[ref_col>1].cpu().tolist())
        
        # Store layer-wise similarities for visualization
        for layer_num, sims in layer_grads.items():
            if sims:
                if layer_num not in layer_wise_stats[label]:
                    layer_wise_stats[label][layer_num] = []
                layer_wise_stats[label][layer_num].append(np.mean(sims))
        
        # Calculate layer differences focusing on early layers (0-10)
        layer_diff_features = []
        early_layer_threshold = 10
        
        # Collect early layers
        early_layers = []
        very_early = []
        mid_early = []
        
        for layer_num, sims in layer_grads.items():
            if layer_num <= early_layer_threshold and sims:
                early_layers.extend(sims)
                if layer_num <= 5:
                    very_early.extend(sims)
                elif 5 < layer_num <= 10:
                    mid_early.extend(sims)
        
        # Calculate features
        if early_layers:
            # [Mean] calculattion form early layers
            layer_diff_features.append(np.mean(early_layers))
            
            # [Difference between very_early, mid_early]
            if very_early and mid_early:
                layer_diff_features.append(abs(np.mean(very_early) - np.mean(mid_early)))
        
        # If no early layer features, use default
        if not layer_diff_features:
            layer_diff_features = [0]
        
        cos_all.append(cos)
        layer_diff_all.append(layer_diff_features)

    # Combine cosine similarity and layer-wise differences
    combined_scores = []
    for i in range(len(cos_all)):
        # Average cosine similarity
        cos_score = sum(cos_all[i])/len(cos_all[i]) if cos_all[i] else 0
        
        # Average layer difference
        layer_diff_score = sum(layer_diff_all[i])/len(layer_diff_all[i]) if layer_diff_all[i] else 0
        
        # Combine scores with weights
        beta = 0.3  
        combined_score = cos_score + beta * layer_diff_score
        combined_scores.append(combined_score)

    # Plotting section
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Precision-Recall Curve
    plt.subplot(2, 2, 1)
    precision, recall, thresholds = precision_recall_curve(label_all, combined_scores)
    auprc = auc(recall, precision)
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.fill_between(recall, precision, alpha=0.3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUPRC: {auprc:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Score Distribution
    plt.subplot(2, 2, 2)
    toxic_scores = [combined_scores[i] for i in range(len(label_all)) if label_all[i] == 1]
    non_toxic_scores = [combined_scores[i] for i in range(len(label_all)) if label_all[i] == 0]
    
    plt.violinplot([toxic_scores], positions=[1], showmeans=True, showextrema=True)
    plt.violinplot([non_toxic_scores], positions=[2], showmeans=True, showextrema=True)
    plt.xticks([1, 2], ['Toxic', 'Non-toxic'])
    plt.ylabel('Combined Score')
    plt.title('Score Distribution by Query Type')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Threshold Analysis
    plt.subplot(2, 2, 3)
    thresholds_test = np.linspace(0.05, 0.5, 50)
    f1_scores = []
    precisions = []
    recalls = []
    
    for thresh in thresholds_test:
        pred = [1 if score >= thresh else 0 for score in combined_scores]
        f1_scores.append(f1_score(label_all, pred, zero_division=0))
        precisions.append(precision_score(label_all, pred, zero_division=0))
        recalls.append(recall_score(label_all, pred, zero_division=0))
    
    # Find optimal threshold
    best_threshold = thresholds_test[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    
    plt.plot(thresholds_test, f1_scores, 'g-', label='F1 Score', linewidth=2)
    plt.plot(thresholds_test, precisions, 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds_test, recalls, 'r-', label='Recall', linewidth=2)
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Early Layer Analysis (0-10)
    plt.subplot(2, 2, 4)
    early_layers_range = range(11)  # 0-10
    
    toxic_early_means = []
    non_toxic_early_means = []
    
    for layer in early_layers_range:
        if layer in layer_wise_stats[1] and layer_wise_stats[1][layer]:
            toxic_early_means.append(np.mean(layer_wise_stats[1][layer]))
        else:
            toxic_early_means.append(np.nan)
            
        if layer in layer_wise_stats[0] and layer_wise_stats[0][layer]:
            non_toxic_early_means.append(np.mean(layer_wise_stats[0][layer]))
        else:
            non_toxic_early_means.append(np.nan)
    
    plt.plot(early_layers_range, toxic_early_means, 'r-o', label='Toxic', linewidth=2, markersize=6)
    plt.plot(early_layers_range, non_toxic_early_means, 'b-o', label='Non-toxic', linewidth=2, markersize=6)
    plt.xlabel('Layer Number')
    plt.ylabel('Average Cosine Similarity')
    plt.title('Early Layer (0-10) Similarity Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'early_layer_analysis_{model_id.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate final metrics with best threshold
    predicted_labels = [1 if score >= best_threshold else 0 for score in combined_scores]
    precision = precision_score(label_all, predicted_labels)
    recall = recall_score(label_all, predicted_labels)
    f1 = f1_score(label_all, predicted_labels)
    
    print(f"Best threshold: {best_threshold}")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUPRC:", auprc)
    
    del gradient_norms_compare
    del cos_all
    del label_all
    del layer_diff_all
    
    return auprc, f1

if __name__ == "__main__":
    for model_id in ['./model/Llama-2-7b-chat-hf']:
        gradient_norms_compare, minus_row_cos, minus_col_cos = find_critical_para(model_id)
        df = pd.read_csv('./data/xstest/xstest_v2_prompts.csv')
        auprc, f1 = cos_sim_xstest(model_id, df, gradient_norms_compare, minus_row_cos, minus_col_cos)