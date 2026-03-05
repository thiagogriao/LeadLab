"""
Funções utilitárias compartilhadas entre os classificadores V1 e V2.
"""
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, accuracy_score, f1_score,
    roc_auc_score, auc, balanced_accuracy_score
)


def encontrar_threshold_otimo(y_true, y_prob):
    """
    Encontra o threshold que maximiza o F1-Score usando a curva Precision-Recall.
    
    Args:
        y_true: Labels reais (0 ou 1)
        y_prob: Probabilidades preditas para classe 1
    
    Returns:
        tuple: (threshold_otimo, melhor_f1)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    idx_melhor = np.argmax(f1_scores[:-1])
    return thresholds[idx_melhor], f1_scores[idx_melhor]


def avaliar_modelo(nome, y_test, y_prob, threshold=None):
    """
    Avalia um modelo e retorna todas as métricas em um dicionário.
    
    Args:
        nome: Nome do modelo para identificação
        y_test: Labels reais
        y_prob: Probabilidades preditas
        threshold: Threshold de decisão (se None, usa o F1 ótimo)
    
    Returns:
        dict: Métricas do modelo
    """
    if threshold is None:
        threshold, _ = encontrar_threshold_otimo(y_test, y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    return {
        'nome': nome,
        'threshold': threshold,
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob),
        'auc_pr': auc(*precision_recall_curve(y_test, y_prob)[1::-1]),
        'y_pred': y_pred,
        'y_prob': y_prob
    }
