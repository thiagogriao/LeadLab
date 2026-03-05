"""
Classificador de Leads V3 — Deep Learning (PyTorch)

Treina uma arquitetura Tabular MLP focada em Embedding de features categóricas.
Usada para modelagem de conversão de Leads com features extraídas.

Uso:
    python 04_v3_deep_learning_pytorch/classificador_leads_v3_final.py
"""
import pandas as pd
import numpy as np
import warnings
import time as _time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    LEADS_PARQUET, MODELO_V3,
    CATEGORICAL_FEATURES_V2, NUMERIC_FEATURES_V2, BINARY_FEATURES_V2, TARGET
)
from src.features import criar_features_avancadas
from src.utils import avaliar_modelo

# =====================================================================
# MODELO PYTORCH (TABULAR MLP COM EMBEDDINGS)
# =====================================================================
class TabularMLP(nn.Module):
    def __init__(self, embedding_sizes, num_numerical_cols, emb_dropout_p=0.1, hidden_sizes=[256, 128, 64], dropout_p=0.3):
        super(TabularMLP, self).__init__()
        
        # Cria as camadas de embedding para cada variável categórica
        # embedding_sizes é uma lista de tuplas: (num_categorias_unicas, dimensao_do_embedding)
        self.embeds = nn.ModuleList([
            nn.Embedding(num_categories, emb_size) for num_categories, emb_size in embedding_sizes
        ])
        
        self.emb_dropout = nn.Dropout(emb_dropout_p)
        self.bn_num = nn.BatchNorm1d(num_numerical_cols)
        
        # Calcula o tamanho da entrada para a primeira camada linear
        num_emb_cols = sum([emb_size for _, emb_size in embedding_sizes])
        input_size = num_emb_cols + num_numerical_cols
        
        # Constrói as camadas ocultas (MLP)
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Mish()) # Função de ativação moderna e suave
            layers.append(nn.Dropout(dropout_p))
            input_size = hidden_size
            
        # Camada de saída (1 neurônio com Sigmoid para probabilidade)
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Flatten()) # flatten out to match labels shape
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x_cat, x_num):
        # 1. Processa as entradas categóricas pelos embeddings
        # x_cat shape: (Batch Size, Qtd. Vars Categóricas)
        if len(self.embeds) > 0 and x_cat is not None:
            emb_outputs = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeds)]
            x_cat_emb = torch.cat(emb_outputs, dim=1) # Concatena todos os embeddings
            x_cat_emb = self.emb_dropout(x_cat_emb)
        else:
            x_cat_emb = torch.empty(x_cat.size(0), 0, device=x_cat.device)
            
        # 2. Processa as entradas numéricas (aplica batch norm pra estabilidade fina)
        if x_num is not None and x_num.shape[1] > 0:
            x_num = self.bn_num(x_num)
        else:
            x_num = torch.empty(x_num.size(0), 0, device=x_num.device)
            
        # 3. Junta tudo e passa pelo MLP (Rede Neural Densa)
        x_combined = torch.cat([x_cat_emb, x_num], dim=1)
        out = self.mlp(x_combined)
        return out

# =====================================================================
# DATASET CUSTOMIZADO DO PYTORCH
# =====================================================================
class LeadsDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        # Converte DataFrames ou Numpy para Tensores do PyTorch
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]

# =====================================================================
# PIPELINE DE PRÉ-PROCESSAMENTO PARA DEEP LEARNING
# =====================================================================
def preparar_dados_dl(df, categorical_cols, numerical_cols):
    """
    Transforma dados Pandas numa estrutura pronta para PyTorch.
    1. LabelEncoding para embbedings (converte strings para índices 0, 1, 2...).
    2. StandardScaler obrigatório para redes neurais (Normaliza float).
    """
    print("    Aplicando Label Encoding em categóricas e StandardScaler em numéricas...")
    label_encoders = {}
    embedding_sizes = []
    
    # Processa Categóricas
    df_cat = np.zeros((len(df), len(categorical_cols)), dtype=np.int64)
    for i, col in enumerate(categorical_cols):
        le = LabelEncoder()
        # Tratamento de Nulos temporário
        valores = df[col].astype(str).fillna("Desconhecido").values 
        df_cat[:, i] = le.fit_transform(valores)
        label_encoders[col] = le
        
        # Regra de ouro para tamanho de embedding: min(50, metade da qtd. de categorias únicas)
        num_classes = len(le.classes_)
        emb_dim = min(50, max(2, num_classes // 2))
        embedding_sizes.append((num_classes, emb_dim))
        
    # Processa Numéricas
    scaler = StandardScaler()
    df_num = scaler.fit_transform(df[numerical_cols].fillna(0))
    
    return df_cat, df_num, label_encoders, scaler, embedding_sizes

# =====================================================================
# EXECUÇÃO PRINCIPAL DO TREINO
# =====================================================================
def main():
    print("=" * 70)
    print(" CLASSIFICADOR DE LEADS V3 — PyTorch (DEEP LEARNING)")
    print("=" * 70)
    print("\n  Modelo: Tabular Multi-Layer Perceptron (MLP)")
    print("  Features: Categorical Embeddings + Numéricas (Batch Normalized)\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Acelerador Detectado: {device.type.upper()}")

    # =====================================================================
    # 1. CARREGAMENTO E ENGENHARIA
    # =====================================================================
    print("\n[*] 1. Carregando base de dados e processando features...")
    try:
        df = pd.read_parquet(LEADS_PARQUET)
    except FileNotFoundError:
        print(f"[!] Erro: '{LEADS_PARQUET}' não encontrado. Rode o gerador primeiro.")
        return

    # ETAPA 1.1 - Limpar colunas legadas para evitar leakage.
    colunas_legadas = ['Score_Oculto_Probabilidade', 'Fator_Sorte']
    df = df.drop(columns=[c for c in colunas_legadas if c in df.columns], errors='ignore')

    df = criar_features_avancadas(df)

    categorical_features = CATEGORICAL_FEATURES_V2
    numeric_features = NUMERIC_FEATURES_V2 + BINARY_FEATURES_V2 # Binárias tratadas como float no DL

    labels = df[TARGET].values
    
    # Calculo de pesos para a função de custo (lida com o desbalanceamento sem precisar de SMOTE)
    pos_weight = (len(labels) - labels.sum()) / max(labels.sum(), 1)
    print(f"    Sinalizador de Peso Positivo (Loss): {pos_weight:.2f}")

    # =====================================================================
    # 2. PRÉ-PROCESSAMENTO PYTORCH
    # =====================================================================
    print("\n[*] 2. Preparando tensores...")
    X_cat_all, X_num_all, encoders, scaler, emb_sizes = preparar_dados_dl(
        df, categorical_features, numeric_features
    )

    # Divisão (Treino 80% / Teste 20%) e logo extração de Validação do Treino
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    train_main_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42, stratify=labels[train_idx])

    # Construção de PyTorch Datasets
    train_ds = LeadsDataset(X_cat_all[train_main_idx], X_num_all[train_main_idx], labels[train_main_idx])
    val_ds   = LeadsDataset(X_cat_all[val_idx], X_num_all[val_idx], labels[val_idx])
    test_ds  = LeadsDataset(X_cat_all[test_idx], X_num_all[test_idx], labels[test_idx])

    # Criação dos DataLoaders (Geradores de Batches otimizados para GPU)
    batch_size = 4096
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"    Configuração da Rede:")
    print(f"      - Embeddings: {emb_sizes}")
    print(f"      - Batch Size: {batch_size}")
    
    # =====================================================================
    # 3. CRIAÇÃO DA REDE E OTIMIZADORES
    # =====================================================================
    model = TabularMLP(
        embedding_sizes=emb_sizes, 
        num_numerical_cols=len(numeric_features),
        hidden_sizes=[512, 256, 128], # Arquitetura robusta
        dropout_p=0.2
    ).to(device)

    # BCEWithLogitsLoss usa ativação Sigmoid *internamente* (melhor estabilidade)
    # Aplicando pos_weight para focarmos nos 10% que compram.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # AdamW
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # =====================================================================
    # 4. LOOP DE TREINAMENTO (ÉPOCAS)
    # =====================================================================
    print("\n[*] 4. Iniciando treinamento Deep Learning...")
    epochs = 50
    patience = 7
    best_val_auc = 0.0
    epochs_no_improve = 0
    _t_inicio = _time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_cat, batch_num, batch_y in train_loader:
            batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
            batch_y = batch_y.unsqueeze(1) # ALINHAR O SHAPE COM OUTPUT (B, 1)
            
            optimizer.zero_grad()
            outputs = model(batch_cat, batch_num)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_y.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # --- Validação ---
        model.eval()
        val_loss = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch_cat, batch_num, batch_y in val_loader:
                batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
                batch_y_unsqueezed = batch_y.unsqueeze(1)
                outputs = model(batch_cat, batch_num)
                loss = criterion(outputs, batch_y_unsqueezed)
                val_loss += loss.item() * batch_y.size(0)
                
                # Transformar logits em probabilidades (Sigmoid)
                probs = torch.sigmoid(outputs).cpu().numpy().ravel()
                val_preds.extend(probs)
                val_trues.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(val_trues, val_preds)
        
        print(f"    Época {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC-ROC: {val_auc:.4f}")
        
        scheduler.step(val_auc) # Monitorar AUC para reduzir a taxa de aprendizado se parar de melhorar
        
        # Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            # Salvar os melhores pesos
            torch.save(model.state_dict(), MODELO_V3)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    -> Early Stopping ativado (Sem melhoria de Val AUC por {patience} épocas).")
                break

    _t_total = _time.time() - _t_inicio
    print(f"\n    [+] Treinamento concluído em {_t_total:.1f} segundos.")

    # =====================================================================
    # 5. AVALIAÇÃO FINAL (CONJUNTO DE TESTE)
    # =====================================================================
    print("\n[*] 5. Avaliando melhor modelo no conjunto de teste...")
    
    # Restaura pesos da melhor época
    model.load_state_dict(torch.load(MODELO_V3, weights_only=True))
    model.eval()
    
    test_preds, test_trues = [], []
    with torch.no_grad():
         for batch_cat, batch_num, batch_y in test_loader:
             batch_cat, batch_num = batch_cat.to(device), batch_num.to(device)
             outputs = model(batch_cat, batch_num)
             probs = torch.sigmoid(outputs).cpu().numpy().ravel()
             test_preds.extend(probs)
             test_trues.extend(batch_y.numpy())
             
    y_test = np.array(test_trues)
    y_prob = np.array(test_preds).ravel()

    # ETAPA 5.1 - Calcular metricas e threshold otimizado.
    res_dl = avaliar_modelo('Tabular MLP', y_test, y_prob)
    threshold_otimo = res_dl['threshold']
    
    print("\n" + "=" * 70)
    print(" RESULTADOS V3 (PyTorch)")
    print("=" * 70)
    print(f"    Threshold F1:         {threshold_otimo:.4f}")
    print(f"    AUC-ROC:              {res_dl['auc_roc']:.4f}")
    print(f"    F1 Score:             {res_dl['f1']:.4f}")
    print(f"    Acurácia Balanceada:  {res_dl['balanced_accuracy']*100:.2f}%")
    print(f"    Acurácia Absoluta:    {res_dl['accuracy']*100:.2f}%")
    
    print(f"\n--- Classification Report (Threshold {threshold_otimo:.4f}) ---")
    y_pred_opt = (y_prob >= threshold_otimo).astype(int)
    print(classification_report(y_test, y_pred_opt))
    
    cm = confusion_matrix(y_test, y_pred_opt)
    print("Matriz de Confusão:")
    print(pd.DataFrame(cm,
        columns=["Previsto: Ñ Compra", "Previsto: Compra"],
        index=["Real: Ñ Compra", "Real: Compra"]
    ))

    # =====================================================================
    # 6. SALVAR ARTEFATOS FINAIS
    # =====================================================================
    # O Pytorch State Dict já foi salvo, mas precisamos salvar as variáveis de transformação (Preprocessors)
    joblib_path = MODELO_V3.replace('.pth', '_pipeline.pkl')
    joblib.dump({
        'categorical_encoders': encoders,
        'scaler': scaler,
        'categorical_features': categorical_features,
        'numeric_features': numeric_features,
        'embedding_sizes': emb_sizes,
        'threshold': threshold_otimo,
        'metricas': {
             'f1_score': res_dl['f1'],
             'auc_roc': res_dl['auc_roc']
        }
    }, joblib_path)
    print(f"\n[+] Pesos salvos em: '{MODELO_V3}'")
    print(f"[+] Pipeline de conversões salva em: '{joblib_path}'")

if __name__ == "__main__":
    main()
