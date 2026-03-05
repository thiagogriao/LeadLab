import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os
import plotly.graph_objects as go
from datetime import datetime
import torch
import torch.nn as nn

# Adiciona o diretório raiz do projeto para importar os módulos do `src`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODELO_V1, MODELO_V2, MODELO_V3,
    CATEGORICAL_FEATURES_BASE, NUMERIC_FEATURES_BASE, BINARY_FEATURES_BASE,
    CATEGORICAL_FEATURES_V2, NUMERIC_FEATURES_V2, BINARY_FEATURES_V2
)
from src.features import criar_features_avancadas, criar_features_base

# =====================================================================
# CONFIGURAÇÃO DE PÁGINA E ESTILOS
# =====================================================================
st.set_page_config(
    page_title="LeadLab CRM | IA de Vendas",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Customizado para deixar o app com cara de SaaS
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 30px; }
    .metric-card { background-color: #F8FAFC; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; }
    .stButton>button { width: 100%; border-radius: 6px; height: 50px; font-weight: 600; }
    .badge-quente { background-color: #DEF7EC; color: #03543F; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-frio { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# MODELO PYTORCH (TABULAR MLP)
# =====================================================================
class TabularMLP(nn.Module):
    def __init__(self, embedding_sizes, num_numerical_cols, emb_dropout_p=0.1, hidden_sizes=[256, 128, 64], dropout_p=0.3):
        super(TabularMLP, self).__init__()
        
        self.embeds = nn.ModuleList([
            nn.Embedding(num_categories, emb_size) for num_categories, emb_size in embedding_sizes
        ])
        
        self.emb_dropout = nn.Dropout(emb_dropout_p)
        self.bn_num = nn.BatchNorm1d(num_numerical_cols)
        
        num_emb_cols = sum([emb_size for _, emb_size in embedding_sizes])
        input_size = num_emb_cols + num_numerical_cols
        
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Mish()) 
            layers.append(nn.Dropout(dropout_p))
            input_size = hidden_size
            
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Flatten()) 
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x_cat, x_num):
        if len(self.embeds) > 0 and x_cat is not None:
            emb_outputs = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeds)]
            x_cat_emb = torch.cat(emb_outputs, dim=1) 
            x_cat_emb = self.emb_dropout(x_cat_emb)
        else:
            x_cat_emb = torch.empty(x_cat.size(0), 0, device=x_cat.device)
            
        if x_num is not None and x_num.shape[1] > 0:
            x_num = self.bn_num(x_num)
        else:
            x_num = torch.empty(x_num.size(0), 0, device=x_num.device)
            
        x_combined = torch.cat([x_cat_emb, x_num], dim=1)
        out = self.mlp(x_combined)
        return out

# =====================================================================
# CACHE E CARREGAMENTO DE MODELOS (CÉREBRO)
# =====================================================================
@st.cache_resource(show_spinner="Carregando Inteligência Artificial...")
def carregar_modelo(versao="V2"):
    """ETAPA 1 - Carrega artefatos do modelo escolhido para inferencia."""
    try:
        if versao == "V3 (Deep Learning / PyTorch)":
            joblib_path = MODELO_V3.replace('.pth', '_pipeline.pkl')
            pipeline = joblib.load(joblib_path)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = TabularMLP(
                embedding_sizes=pipeline['embedding_sizes'], 
                num_numerical_cols=len(pipeline['numeric_features']),
                hidden_sizes=[512, 256, 128],
                dropout_p=0.2
            ).to(device)
            model.load_state_dict(torch.load(MODELO_V3, map_location=device, weights_only=True))
            model.eval()
            
            pipeline_ia = {
                'model': model,
                'pipeline': pipeline,
                'device': device,
                'threshold': pipeline['threshold']
            }
            return pipeline_ia, "V3"
        elif versao == "V2 (Ensemble Avançado)":
            return joblib.load(MODELO_V2), "V2"
        else:
            return joblib.load(MODELO_V1), "V1"
    except FileNotFoundError:
        return None, None

# =====================================================================
# FUNÇÕES CORE DE PREDIÇÃO
# =====================================================================
def prever_leads(df_bruto, pipeline_ia, versao):
    """
    ETAPA 2 - Prepara dados de entrada e executa inferencia.
    Retorna probabilidades e threshold do modelo selecionado.
    """
    # ETAPA 2.1 - Copia defensiva e feature engineering por versao.
    df_proc = df_bruto.copy()
    
    if versao == "V3":
        df_proc = criar_features_avancadas(df_proc)
        pipeline = pipeline_ia['pipeline']
        categorical_cols = pipeline['categorical_features']
        numerical_cols = pipeline['numeric_features']
        label_encoders = pipeline['categorical_encoders']
        scaler = pipeline['scaler']
        
        # ETAPA 2.2 - Garantir schema minimo esperado pelo pipeline salvo.
        for col in categorical_cols + numerical_cols:
            if col not in df_proc.columns:
                if col in categorical_cols:
                    df_proc[col] = "Desconhecido"
                else:
                    df_proc[col] = 0
                    
        df_cat = np.zeros((len(df_proc), len(categorical_cols)), dtype=np.int64)
        for i, col in enumerate(categorical_cols):
            le = label_encoders[col]
            valores = df_proc[col].astype(str).fillna("Desconhecido").values
            known_classes = set(le.classes_)
            mapped_valores = []
            for v in valores:
                if v in known_classes:
                    mapped_valores.append(v)
                elif "Desconhecido" in known_classes:
                    mapped_valores.append("Desconhecido")
                else:
                    mapped_valores.append(le.classes_[0])
            df_cat[:, i] = le.transform(mapped_valores)
            
        df_num = scaler.transform(df_proc[numerical_cols].fillna(0))
        
        device = pipeline_ia['device']
        model = pipeline_ia['model']
        
        X_cat = torch.tensor(df_cat, dtype=torch.long).to(device)
        X_num = torch.tensor(df_num, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(X_cat, X_num)
            probabilidades = torch.sigmoid(outputs).cpu().numpy().flatten()
            
        threshold = pipeline_ia['threshold']
        
        return probabilidades, threshold

    elif versao == "V2":
        df_proc = criar_features_avancadas(df_proc)
        features_necessarias = CATEGORICAL_FEATURES_V2 + NUMERIC_FEATURES_V2 + BINARY_FEATURES_V2
    else:
        df_proc = criar_features_base(df_proc)
        features_necessarias = CATEGORICAL_FEATURES_BASE + NUMERIC_FEATURES_BASE + BINARY_FEATURES_BASE
        
    # ETAPA 2.2 - Garantir schema minimo esperado pelo pipeline sklearn.
    for col in features_necessarias:
        if col not in df_proc.columns:
            if col in CATEGORICAL_FEATURES_V2:
                df_proc[col] = "Desconhecido"
            else:
                df_proc[col] = 0

    X_inferencia = df_proc[features_necessarias]
    
    # ETAPA 2.3 - Transformar entrada e rodar preditores.
    X_transformado = pipeline_ia['preprocessor'].transform(X_inferencia)
    
    # ETAPA 2.4 - Combinar saidas conforme arquitetura selecionada.
    if versao == "V2":
        prob_xgb = pipeline_ia['xgb_model'].predict_proba(X_transformado)[:, 1]
        prob_lgb = pipeline_ia['lgb_model'].predict_proba(X_transformado)[:, 1]
        prob_cb  = pipeline_ia['cb_model'].predict_proba(X_transformado)[:, 1]
        
        X_meta = np.column_stack([prob_xgb, prob_lgb, prob_cb])
        probabilidades = pipeline_ia['meta_learner'].predict_proba(X_meta)[:, 1]
    else:
        probabilidades = pipeline_ia['model'].predict_proba(X_transformado)[:, 1]
        
    threshold = pipeline_ia['threshold']
    
    return probabilidades, threshold

# =====================================================================
# MENU LATERAL E CABEÇALHO
# =====================================================================
st.markdown('<p class="main-header">🎯 LeadLab CRM</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Plataforma de IA Preditiva para Otimização de Funil de Vendas</p>', unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1453/1453008.png", width=60)
st.sidebar.markdown("### Configurações do Motor")

motor_escolhido = st.sidebar.selectbox(
    "1. Selecione a Inteligência",
    ["V3 (Deep Learning / PyTorch)", "V2 (Ensemble Avançado)", "V1 (XGBoost Baseline)"],
    help="O Ensemble usa XGBoost + LightGBM + CatBoost juntos para maior precisão. A V3 usa uma rede neural PyTorch com Embeddings."
)

st.sidebar.markdown("---")
# st.sidebar.info("A versão **V3 (Deep Learning / PyTorch)** requer instâncias com *GPU Cuda* para rodar a inferência dos tensores em lote. Exibindo apenas modelos *CPU-Bound*.")

# Carrega os cérebros na Memória RAM do servidor Web
modelo_pipeline, versao_motor = carregar_modelo(motor_escolhido)

if not modelo_pipeline:
    st.error(f"⚠️ O Cérebro da Inteligência ({motor_escolhido}) não foi encontrado na pasta `models/`.\nPor favor, treine o modelo no laboratório primeiro.")
    st.stop()

# =====================================================================
# ABAS DE NAVEGAÇÃO DA PLATAFORMA (SPA)
# =====================================================================
aba_simulador, aba_lote = st.tabs([
    "👨‍💼 Simulador Individual (Lead Call)", 
    "📁 Bateira de Lote (Marketing Batch)"
])

# ---------------------------------------------------------------------
# ABA 1: SIMULADOR DE LEAD ÚNICO (CRM)
# ---------------------------------------------------------------------
with aba_simulador:
    st.markdown("### Simulador de Atendimento ao Vivo")
    st.write("Preencha os dados do Lead enquanto está em ligação ou prospectando para receber a Temperatura da Venda.")
    
    with st.form("form_lead"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Dados Demográficos**")
            idade = st.number_input("Idade", min_value=18, max_value=100, value=30)
            salario = st.number_input("Renda Estimada (R$)", min_value=0, value=5000, step=1000)
            estado_civil = st.selectbox("Estado Civil", ["Solteiro(a)", "Casado(a)", "Divorciado(a)", "Viuvo(a)"])
            genero = st.selectbox("Gênero", ["M", "F", "Outro"])
            
        with col2:
            st.markdown("**Perfil Corporativo**")
            cargo = st.selectbox("Cargo", ["Analista", "Gerente", "Diretor", "C-Level", "Especialista"])
            setor = st.selectbox("Setor", ["Tecnologia", "Varejo", "Saude", "Educacao", "Servicos", "Bancario"])
            tamanho = st.selectbox("Tamanho da Empresa", ["1-10", "11-50", "51-200", "201-500", "501+"])
            email_corp = st.checkbox("Possui E-mail Corporativo?", value=True)
            
        with col3:
            st.markdown("**Pegada Digital (Interesse)**")
            origem = st.selectbox("Origem do Lead", ["Google Ads", "Instagram", "LinkedIn", "Acesso Direto", "Busca Organica", "Email Marketing", "TikTok"])
            paginas = st.slider("Páginas Visitadas", 1, 50, 3)
            tempo_site = st.slider("Minutos no Site", 1, 60, 5)
            abriu_email = st.checkbox("Abriu nosso E-mail?", value=False)
            clicou_email = st.checkbox("Clicou no Link da Oferta?", value=False)
            
        st.markdown("<br>", unsafe_allow_html=True)
        submeteu_lead = st.form_submit_button("🔥 Calcular Propensão de Fechamento", use_container_width=True)

    if submeteu_lead:
        # Montar um DataFrame falso com os dados do Formulário
        df_lead = pd.DataFrame([{
            'Idade': idade,
            'Salario': salario,
            'Estado_Civil': estado_civil,
            'Genero': genero,
            'Cargo': cargo,
            'Setor': setor,
            'Tamanho_Empresa': tamanho,
            'Email_Corporativo': "email@empresa.com" if email_corp else "",
            'Origem': origem,
            'Paginas_Visitadas': paginas,
            'Tempo_Site_Min': tempo_site,
            'Abriu_Email': int(abriu_email),
            'Clicou_Email': int(clicou_email),
            # Preenchimento de campos obrigáveis de estado base
            'Data_Nascimento': datetime(datetime.now().year - idade, 1, 1).strftime('%Y-%m-%d'),
            'Estado': 'SP', 'Dispositivo': 'Desktop', 'Sistema_Operacional': 'Windows'
        }])
        
        with st.spinner("Analisando com a Inteligência Artificial..."):
            probs, threshold = prever_leads(df_lead, modelo_pipeline, versao_motor)
            prob_venda = probs[0]
            
        st.markdown("---")
        
        col_grafico, col_texto = st.columns([1, 1])
        
        with col_grafico:
            # Velocímetro Visual Clássico
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_venda * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Temperatura do Lead"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, threshold * 100], 'color': "#FDE8E8"}, # Vermelho Dificil
                        {'range': [threshold * 100, (threshold+0.2)*100], 'color': "#FEF4C6"}, # Amarelo Médio
                        {'range': [(threshold+0.2)*100, 100], 'color': "#DEF7EC"} # Verde Quente
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
        with col_texto:
            st.markdown("### O Veredito da Inteligência")
            if prob_venda >= threshold:
                st.success(f"**🟢 PROSSIGA: Lead Quente!**\nA probabilidade de fechamento ({prob_venda*100:.1f}%) superou a nota de corte do mercado rigoroso ({threshold*100:.1f}%). Dedique tempo de venda a ele.")
                st.balloons()
            else:
                st.error(f"**🔴 ABORTE: Lead Frio.**\nA probabilidade de apenas {prob_venda*100:.1f}% indica que ele tem perfil de 'Turista' ou baixa aptidão financeira. Automatize via Marketing e não gaste telefone.")
            
            # Box de explicações estáticas (Simulando uma SHAP interpretability simples)
            st.markdown("""
            **Fatores Favoráveis Potenciais:**
            - Presença de E-mail Corporativo ou cliques recentes na Newsletter.
            - Tempo elevado navegando no Produto (Intensidade de Rastreio).
            """)

# ---------------------------------------------------------------------
# ABA 2: PROCESSAMENTO EM LOTE (BATCH MARKETING)
# ---------------------------------------------------------------------
with aba_lote:
    st.markdown("### Processamento Massivo de Listas")
    st.write("Exporte um arquivo final (Parquet ou CSV) do seu sistema para limpar os Leads inúteis antes de enviar para o setor de Venda Ativa.")
    
    st.info("💡 **Dica Didática LeadLab:** Como este é o nosso Laboratório, sinta-se à vontade para enviar o próprio arquivo `data/leads_gerados.parquet` gerado no Módulo 01 para ver a IA peneirar 1 Milhão de registros em segundos.")
    
    arquivo_upload = st.file_uploader("Arraste seu arquivo de Leads aqui (.csv, .parquet)", type=['csv', 'parquet'])
    
    if arquivo_upload is not None:
        try:
            if arquivo_upload.name.endswith('.csv'):
                df_lote = pd.read_csv(arquivo_upload)
            else:
                df_lote = pd.read_parquet(arquivo_upload)
                
            st.success(f"Arquivo carregado com sucesso! Linhas detectadas: {len(df_lote):,}")
            
            if st.button("🚀 Processar Classificação com IA", type="primary"):
                with st.spinner(f"O Cérebro {versao_motor} está analisando {len(df_lote):,} registros simultaneamente. Aguarde..."):
                    
                    # Remover alvos legados (Caso usem a base oficial gerada por nós que já tem o gabarito)
                    df_lote_limpo = df_lote.drop(columns=['Status_Venda', 'Score_Oculto_Probabilidade', 'Fator_Sorte'], errors='ignore')
                    
                    probs_lote, thresh_lote = prever_leads(df_lote_limpo, modelo_pipeline, versao_motor)
                    
                    # Anexar as Respostas de Banco
                    df_resultado = df_lote.copy()
                    df_resultado['IA_Probabilidade_%'] = np.round(probs_lote * 100, 2)
                    df_resultado['IA_Classificacao'] = np.where(probs_lote >= thresh_lote, 'QUENTE (Abordar)', 'FRIO (Descartar)')
                    
                    # ===============================================
                    # DASHBOARD DE FUNIL DO LOTE
                    # ===============================================
                    qtd_quente = (probs_lote >= thresh_lote).sum()
                    qtd_frio = len(probs_lote) - qtd_quente
                    
                    st.markdown("### 📊 Relatório Gerencial da Base")
                    
                    colA, colB, colC = st.columns(3)
                    colA.metric("Total Analisado", f"{len(df_resultado):,}")
                    colB.metric("🟢 Qualificados (Quentes)", f"{qtd_quente:,}", f"{(qtd_quente/len(df_resultado))*100:.1f}% da base")
                    colC.metric("🔴 Descartados (Frios)", f"{qtd_frio:,}", f"-{(qtd_frio/len(df_resultado))*100:.1f}% de Lixo")
                    
                    st.markdown("---")
                    
                    # Tabela Interativa de Resultado
                    st.markdown("**Prévia da Planilha Filtrada (Top 100 Quentes)**")
                    st.dataframe(
                        df_resultado.sort_values(by="IA_Probabilidade_%", ascending=False).head(100),
                        use_container_width=True
                    )
                    
                    # Download CSV
                    csv_saida = df_resultado.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Baixar Planilha Enriquecida Completa",
                        data=csv_saida,
                        file_name=f"leads_enriquecidos_{versao_motor}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Erro ao tentar ler ou processar o arquivo: {str(e)}")
