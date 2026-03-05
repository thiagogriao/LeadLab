"""
Configurações centralizadas do projeto LeadLab.
Caminhos, constantes e listas de features compartilhadas entre os scripts.
"""
import os

# =====================================================================
# CAMINHOS DO PROJETO
# =====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CEPS_DIR = os.path.join(BASE_DIR, 'ceps')
ESTUDOS_DIR = os.path.join(DATA_DIR, 'estudos')

# Cria pastas se não existirem
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ESTUDOS_DIR, exist_ok=True)

# Arquivos de dados
LEADS_PARQUET = os.path.join(DATA_DIR, 'leads_gerados.parquet')
CLIENTES_PARQUET = os.path.join(DATA_DIR, 'clientes_gerados.parquet')
CEPS_PARQUET = os.path.join(DATA_DIR, 'ceps_validos.parquet')
BASE_CEPS_PARQUET = os.path.join(CEPS_DIR, 'base_ceps_otimizada.parquet')

# Arquivos de modelos
MODELO_V1 = os.path.join(MODELS_DIR, 'modelo_xgboost_leads_v1.pkl')
MODELO_V2 = os.path.join(MODELS_DIR, 'modelo_ensemble_leads_v2.pkl')
MODELO_V3 = os.path.join(MODELS_DIR, 'modelo_dl_leads_v3.pth')

# =====================================================================
# FEATURES DO MODELO
# =====================================================================

# Features Categóricas (comuns V1 e V2)
CATEGORICAL_FEATURES_BASE = [
    'Genero', 'Estado_Civil', 'Setor', 'Tamanho_Empresa',
    'Estado', 'Origem', 'Dispositivo', 'Sistema_Operacional',
    'Faixa_Salarial'
]

# Features Categóricas extras do V2
CATEGORICAL_FEATURES_V2 = CATEGORICAL_FEATURES_BASE + ['Faixa_Idade']

# Features Numéricas (comuns V1 e V2)
NUMERIC_FEATURES_BASE = [
    'Idade', 'Salario', 'Paginas_Visitadas', 'Tempo_Site_Min',
    'Score_Engajamento', 'Engajamento_x_Clicou', 'Intensidade_Navegacao',
    'Salario_Norm'
]

# Features Numéricas extras do V2 (polinomiais)
NUMERIC_FEATURES_V2 = NUMERIC_FEATURES_BASE + [
    'Engajamento_Quadrado', 'Salario_x_Engajamento', 'Idade_x_Salario'
]

# Features Binárias (comuns V1 e V2)
BINARY_FEATURES_BASE = [
    'Tem_Email_Corp', 'Abriu_Email', 'Clicou_Email'
]

# Features Binárias extras do V2
BINARY_FEATURES_V2 = BINARY_FEATURES_BASE + [
    'Engajamento_Alto', 'Salario_Alto'
]

# Target
TARGET = 'Status_Venda'

# =====================================================================
# CONFIGURAÇÕES DO GERADOR
# =====================================================================
QTD_CEPS_VALIDOS = 10000
QTD_LEADS_TOTAL = 1_000_000
QTD_CLIENTES_ALVO = 100_000
PESO_DETERMINISTICO = 0.95
PESO_ALEATORIO = 0.05
