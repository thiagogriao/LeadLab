"""
Fragmento de estudo (V2): stacking e meta-learning.

Este arquivo mostra como combinar as probabilidades dos modelos base.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def consolidar_opinioes(xgb, lgb, cb, X_dados):
    """
    ETAPA 1 - Gerar probabilidades dos base learners.
    ETAPA 2 - Empilhar em matriz [prob_xgb, prob_lgb, prob_cb].
    """
    prob_xgb = xgb.predict_proba(X_dados)[:, 1]
    prob_lgb = lgb.predict_proba(X_dados)[:, 1]
    prob_cb = cb.predict_proba(X_dados)[:, 1]
    return np.column_stack([prob_xgb, prob_lgb, prob_cb])


def criar_meta_learner():
    """ETAPA 3 - Criar o modelo que aprende a combinar as tres saidas."""
    return LogisticRegression(random_state=42, max_iter=1000, C=1.0)


if __name__ == "__main__":
    print("[*] Stacking da V2:")
    print("    1) gerar probabilidades dos 3 modelos")
    print("    2) empilhar em matriz meta")
    print("    3) treinar LogisticRegression com a validacao")
