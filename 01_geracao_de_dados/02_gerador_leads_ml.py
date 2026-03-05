"""
Gerador de dados sinteticos do LeadLab.

Etapas principais:
1) montar perfil de mercado (B2B/B2C e escopo geografico);
2) carregar/gerar cache de CEPs validos;
3) gerar leads e clientes com score oculto;
4) exportar artefatos parquet para os modulos de modelagem.
"""
import pandas as pd
import numpy as np
from faker import Faker
import random
import time
import os
from typing import Dict, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    LEADS_PARQUET, CLIENTES_PARQUET, CEPS_PARQUET, BASE_CEPS_PARQUET,
    QTD_CEPS_VALIDOS, QTD_LEADS_TOTAL, QTD_CLIENTES_ALVO,
    PESO_DETERMINISTICO, PESO_ALEATORIO
)

# Configuração do Faker para localização brasileira
fake = Faker('pt_BR')

# Caminhos centralizados (via src/config.py)
ARQUIVO_CEPS = CEPS_PARQUET
ARQUIVO_LEADS = LEADS_PARQUET
ARQUIVO_CLIENTES = CLIENTES_PARQUET

# =====================================================================
# CONFIGURAÇÕES GLOBAIS
# =====================================================================
ESTADOS_ALVO_REGIONAIS = ["SP", "RJ", "MG", "PR", "SC", "RS"]


def obter_perfil_mercado(modo_negocio="b2b", escopo_geografico="regional"):
    modo = str(modo_negocio).lower().strip()
    escopo = str(escopo_geografico).lower().strip()

    if modo not in {"b2b", "b2c"}:
        raise ValueError("modo_negocio deve ser 'b2b' ou 'b2c'.")
    if escopo not in {"regional", "nacional"}:
        raise ValueError("escopo_geografico deve ser 'regional' ou 'nacional'.")

    base = {
        "estados_alvo": ESTADOS_ALVO_REGIONAIS if escopo == "regional" else None,
        "generos": ["Masculino", "Feminino"],
        "pesos_genero": [0.485, 0.515],
        "estados_civis": ["Solteiro", "Casado", "Divorciado", "Viúvo"],
        "pesos_estado_civil": [0.30, 0.51, 0.13, 0.06],
        "complementos": ["Apto 12", "Bloco B", "Casa 2", "Sala 100", "N/A"],
        "so_por_dispositivo": {
            "Mobile": (["Android", "iOS"], [0.83, 0.17]),
            "Desktop": (["Windows", "macOS", "Linux"], [0.88, 0.07, 0.05]),
            "Tablet": (["Android", "iOS"], [0.60, 0.40]),
        },
    }

    if modo == "b2b":
        base.update(
            {
                "prob_email_corp": 0.35,
                "setores": ["Tecnologia", "Varejo", "Indústria", "Saúde", "Finanças", "Educação", "Serviços", "E-commerce"],
                "pesos_setor": [0.12, 0.16, 0.12, 0.10, 0.08, 0.09, 0.25, 0.08],
                "tamanhos_empresa": ["1-10", "11-50", "51-200", "201-500", "500+"],
                "pesos_tamanho": [0.55, 0.27, 0.10, 0.05, 0.03],
                "origens": [
                    "Meta Ads", "Instagram Orgânico", "TikTok Ads", "Spotify Ads",
                    "Google Ads", "YouTube Ads", "LinkedIn Ads", "Busca Orgânica/SEO",
                    "Acesso Direto/Site", "E-mail Marketing", "Indicação", "Cold Call",
                    "Feiras/Eventos", "Pinterest Ads"
                ],
                "pesos_origem": [
                    0.15, 0.08, 0.05, 0.02,
                    0.22, 0.05, 0.04, 0.14,
                    0.08, 0.07, 0.05, 0.02,
                    0.02, 0.01
                ],
                "dispositivos": ["Mobile", "Desktop", "Tablet"],
                "pesos_dispositivo": [0.65, 0.30, 0.05],
                "pesos_score": {"engajamento": 0.35, "origem": 0.25, "financeiro": 0.20, "empresa": 0.10, "dispositivo": 0.10},
                "origens_score": {
                    "Indicação": 0.88,
                    "Google Ads": 0.78,
                    "LinkedIn Ads": 0.75,
                    "Busca Orgânica/SEO": 0.68,
                    "E-mail Marketing": 0.62,
                    "Feiras/Eventos": 0.60,
                    "Meta Ads": 0.53,
                    "YouTube Ads": 0.50,
                    "Instagram Orgânico": 0.45,
                    "Pinterest Ads": 0.40,
                    "TikTok Ads": 0.36,
                    "Spotify Ads": 0.31,
                    "Acesso Direto/Site": 0.38,
                    "Cold Call": 0.22,
                },
                "dispositivos_score": {"Desktop": 0.65, "Tablet": 0.48, "Mobile": 0.58},
            }
        )
    else:
        base.update(
            {
                "prob_email_corp": 0.10,
                "setores": ["Varejo", "Serviços", "E-commerce", "Educação", "Saúde", "Tecnologia", "Finanças", "Indústria"],
                "pesos_setor": [0.24, 0.28, 0.18, 0.08, 0.08, 0.06, 0.04, 0.04],
                "tamanhos_empresa": ["1-10", "11-50", "51-200", "201-500", "500+"],
                "pesos_tamanho": [0.72, 0.19, 0.06, 0.02, 0.01],
                "origens": [
                    "Meta Ads", "Instagram Orgânico", "TikTok Ads", "Spotify Ads",
                    "Google Ads", "YouTube Ads", "LinkedIn Ads", "Busca Orgânica/SEO",
                    "Acesso Direto/Site", "E-mail Marketing", "Indicação", "Cold Call",
                    "Feiras/Eventos", "Pinterest Ads"
                ],
                "pesos_origem": [
                    0.20, 0.15, 0.10, 0.03,
                    0.16, 0.07, 0.01, 0.12,
                    0.08, 0.03, 0.03, 0.00,
                    0.01, 0.01
                ],
                "dispositivos": ["Mobile", "Desktop", "Tablet"],
                "pesos_dispositivo": [0.78, 0.18, 0.04],
                "pesos_score": {"engajamento": 0.45, "origem": 0.20, "financeiro": 0.15, "empresa": 0.05, "dispositivo": 0.15},
                "origens_score": {
                    "Indicação": 0.83,
                    "Google Ads": 0.73,
                    "LinkedIn Ads": 0.36,
                    "Busca Orgânica/SEO": 0.66,
                    "E-mail Marketing": 0.55,
                    "Feiras/Eventos": 0.44,
                    "Meta Ads": 0.64,
                    "YouTube Ads": 0.54,
                    "Instagram Orgânico": 0.60,
                    "Pinterest Ads": 0.46,
                    "TikTok Ads": 0.49,
                    "Spotify Ads": 0.32,
                    "Acesso Direto/Site": 0.58,
                    "Cold Call": 0.10,
                },
                "dispositivos_score": {"Desktop": 0.58, "Tablet": 0.48, "Mobile": 0.63},
            }
        )

    return base

# =====================================================================
# FASE A: GARIMPO GEOGRÁFICO DINÂMICO E CACHE
# =====================================================================
def gerar_cache_enderecos(quantidade=100, estados_alvo=None):
    if os.path.exists(ARQUIVO_CEPS):
        print(f"[*] Fase A: Arquivo '{ARQUIVO_CEPS}' encontrado. Carregando CEPs salvos...")
        df_ceps = pd.read_parquet(ARQUIVO_CEPS)
        df_ceps['CEP'] = df_ceps['CEP'].astype(str).str.zfill(8)
        if estados_alvo:
            df_ceps = df_ceps[df_ceps['Estado'].isin(estados_alvo)]
        else:
            # Cache antigo pode estar regionalizado; nesse caso força recarga da base completa.
            if "Estado" in df_ceps.columns and df_ceps["Estado"].nunique() < 20:
                print("  -> Cache local parece regionalizado. Regerando cache nacional a partir da base completa...")
                os.remove(ARQUIVO_CEPS)
                return gerar_cache_enderecos(quantidade=quantidade, estados_alvo=None)

        if len(df_ceps) < quantidade:
            print(f"  -> [Aviso] Cache tem {len(df_ceps)} CEPs após filtro; ajustando quantidade.")
            quantidade = len(df_ceps)
        df_ceps = df_ceps.sample(n=quantidade)
        cache_enderecos = df_ceps.to_dict('records')
        print(f"  -> Foram carregados {len(cache_enderecos)} endereços do cache em disco.\n")
        return cache_enderecos
        
    print(f"[*] Iniciando Fase A: Garimpando {quantidade} endereços na base otimizada (Parquet)...")
    
    arquivo_parquet = BASE_CEPS_PARQUET
    if not os.path.exists(arquivo_parquet):
        raise FileNotFoundError(f"Base de CEPs não encontrada em '{arquivo_parquet}'.")
        
    try:
        df_ceps = pd.read_parquet(arquivo_parquet)
        
        if estados_alvo:
            df_ceps = df_ceps[df_ceps['Estado'].isin(estados_alvo)]
            print(f"  -> Filtro de Estados ativo: Segmentando busca para {estados_alvo}")
            
        if len(df_ceps) < quantidade:
            print(f"  -> [Aviso] A base filtrada tem menos registros ({len(df_ceps)}) que a quantidade solicitada ({quantidade}).")
            quantidade = len(df_ceps)
            
        amostra = df_ceps.sample(n=quantidade)
        
        # Garante a formatação do CEP com zeros à esquerda
        amostra['CEP'] = amostra['CEP'].astype(str).str.zfill(8)
        
        # Opcional: preencher campos em branco (já q o parquet pode ter alguns)
        amostra['Logradouro'] = amostra['Logradouro'].fillna('Centro').replace('', 'Centro')
        amostra['Bairro'] = amostra['Bairro'].fillna('Centro').replace('', 'Centro')
        
        cache_enderecos = amostra.to_dict('records')
            
        pd.DataFrame(cache_enderecos).to_parquet(ARQUIVO_CEPS, index=False, engine='pyarrow')
        print(f"[+] Fase A concluída: {quantidade} Endereços salvos em '{ARQUIVO_CEPS}'.\n")
        return cache_enderecos
        
    except Exception as e:
        raise RuntimeError(f"Erro ao processar base de CEPs em '{arquivo_parquet}': {e}") from e

# =====================================================================
# MOTOR DE PROBABILIDADE DE COMPRA (PADRÕES REALISTAS)
# =====================================================================
def calcular_score_deterministico(lead, perfil_score: Dict[str, Any]):
    """
    Calcula a probabilidade de compra de um lead com base em fatores
    realistas. Retorna um score de 0.0 a 1.0.
    
    Pesos:
      - Engajamento digital:  35%
      - Origem de aquisição:  25%
      - Poder financeiro:     20%
      - Perfil empresarial:   10%
      - Dispositivo/Canal:    10%
    """
    
    # --- 1. SCORE DE ENGAJAMENTO (0 a 1) ---
    # Páginas visitadas: quanto mais visita, maior o interesse (1 a 30)
    pag_score = min(lead["Paginas_Visitadas"] / 20.0, 1.0)
    
    # Tempo no site: mais tempo = mais interesse (0.5 a 45 min)
    tempo_score = min(lead["Tempo_Site_Min"] / 30.0, 1.0)
    
    # E-mail: abriu + clicou = forte sinal de engajamento
    email_score = 0.0
    if lead["Abriu_Email"] == 1:
        email_score = 0.4
    if lead["Clicou_Email"] == 1:
        email_score = 1.0
    
    engajamento = pag_score * 0.3 + tempo_score * 0.3 + email_score * 0.4
    
    # --- 2. SCORE DE ORIGEM (0 a 1) ---
    origens_alta_conversao = perfil_score["origens_score"]
    origem_score = origens_alta_conversao.get(lead["Origem"], 0.3)
    
    # --- 3. SCORE FINANCEIRO (0 a 1) ---
    salario = lead["Salario"]
    if salario >= 20000:
        financeiro = 1.0
    elif salario >= 10000:
        financeiro = 0.75
    elif salario >= 6000:
        financeiro = 0.50
    elif salario >= 3000:
        financeiro = 0.30
    else:
        financeiro = 0.10
    
    # --- 4. SCORE DA EMPRESA (0 a 1) ---
    # Setores mais propensos a compra
    setores_quentes = {
        "Tecnologia": 0.90,
        "E-commerce": 0.85,
        "Finanças": 0.75,
        "Saúde": 0.60,
        "Serviços": 0.50,
        "Indústria": 0.45,
        "Educação": 0.35,
        "Varejo": 0.30,
    }
    setor_score = setores_quentes.get(lead["Setor"], 0.3)
    
    # Tamanho da empresa — maiores têm mais budget
    tamanhos_score = {
        "500+": 0.90,
        "201-500": 0.75,
        "51-200": 0.60,
        "11-50": 0.40,
        "1-10": 0.20,
    }
    tamanho_score = tamanhos_score.get(lead["Tamanho_Empresa"], 0.3)
    
    empresa = setor_score * 0.6 + tamanho_score * 0.4
    
    # --- 5. SCORE DE DISPOSITIVO (0 a 1) ---
    dispositivos_score = perfil_score["dispositivos_score"]
    dispositivo = dispositivos_score.get(lead["Dispositivo"], 0.4)
    
    # --- SCORE FINAL COMPOSTO ---
    pesos = perfil_score["pesos_score"]
    score_deterministico = (
        engajamento * pesos["engajamento"] +
        origem_score * pesos["origem"] +
        financeiro * pesos["financeiro"] +
        empresa * pesos["empresa"] +
        dispositivo * pesos["dispositivo"]
    )
    return score_deterministico


def calcular_probabilidade_compra(lead, perfil_score: Dict[str, Any]):
    score_deterministico = calcular_score_deterministico(lead, perfil_score)
    # Componente aleatório mínimo (5%) — simula imprevisibilidade real sem poluir os padrões
    return score_deterministico * PESO_DETERMINISTICO + random.random() * PESO_ALEATORIO


# =====================================================================
# FASE B: GERAÇÃO DE IDENTIDADES SINTÉTICAS
# =====================================================================
def gerar_dados_sinteticos(
    cache_enderecos,
    num_leads=10000,
    num_clientes=None,
    taxa_conversao=0.05,
    modo_negocio="b2b",
    escopo_geografico="regional",
):
    if os.path.exists(ARQUIVO_LEADS):
        print(f"[*] Fase B: Arquivo '{ARQUIVO_LEADS}' encontrado. Carregando Leads salvos...")
        df_leads = pd.read_parquet(ARQUIVO_LEADS)
        leads = df_leads.to_dict('records')
        print(f"  -> Foram carregados {len(leads)} leads do cache em disco.\n")
        return leads

    perfil = obter_perfil_mercado(modo_negocio=modo_negocio, escopo_geografico=escopo_geografico)
    print(
        f"[*] Iniciando Fase B: Gerando {num_leads} leads sintéticos "
        f"({modo_negocio.upper()} / {escopo_geografico})..."
    )
    leads = []
    
    # =====================================================================
    # DISTRIBUIÇÕES REALISTAS — FONTES: IBGE PNAD 2024, CENSO 2022,
    #                                   STATCOUNTER 2024, SEBRAE
    # =====================================================================
    
    # Gênero — IBGE (48.5% Masc, 51.5% Fem)
    generos = perfil["generos"]
    pesos_genero = perfil["pesos_genero"]
    
    # Estado Civil — Censo 2022 (51% em união, 30% solteiro, 13% divorciado, 6% viúvo)
    estados_civis = perfil["estados_civis"]
    pesos_estado_civil = perfil["pesos_estado_civil"]
    
    # Setores — Composição PIB/emprego Brasil
    # Serviços domina (~70% PIB), mas distribuímos entre sub-setores
    setores = perfil["setores"]
    pesos_setor = perfil["pesos_setor"]
    
    # Tamanho Empresa — SEBRAE (97% micro/pequenas no Brasil)
    tamanhos_empresa = perfil["tamanhos_empresa"]
    pesos_tamanho = perfil["pesos_tamanho"]
    
    complementos = perfil["complementos"]
    
    # Origens de Aquisição — Market Share digital Brasil 2024
    # Google domina busca (~22%), Meta/Instagram ads são fortes, SEO relevante
    origens = perfil["origens"]
    pesos_origem = perfil["pesos_origem"]
    
    # Dispositivo — StatCounter Brasil 2024 (70% Mobile, 25% Desktop, 5% Tablet)
    dispositivos = perfil["dispositivos"]
    pesos_dispositivo = perfil["pesos_dispositivo"]
    
    # SO correlacionado por dispositivo — StatCounter 2024
    so_por_dispositivo = perfil["so_por_dispositivo"]

    for i in range(num_leads):
        # =============================================================
        # [DADOS PESSOAIS] — Distribuições IBGE
        # =============================================================
        genero = np.random.choice(generos, p=pesos_genero)
        if genero == "Masculino":
            nome = fake.name_male()
        else:
            nome = fake.name_female()
        
        estado_civil = np.random.choice(estados_civis, p=pesos_estado_civil)
        
        # Idade — Pirâmide etária brasileira (média 35, desvio 12, truncada 18-70)
        idade_valor = int(np.clip(np.random.normal(35, 12), 18, 70))
        # Gera data de nascimento coerente com a idade
        data_nascimento = fake.date_of_birth(
            minimum_age=idade_valor, maximum_age=idade_valor
        ).strftime("%Y-%m-%d")
            
        # =============================================================
        # [DADOS DE CONTATO] — Email corp. condicional (~35%)
        # =============================================================
        email_pessoal = fake.email()
        # Probabilidade de e-mail corporativo ajustada por modo de negócio
        tem_email_corp = random.random() < perfil["prob_email_corp"]
        email_corporativo = fake.company_email() if tem_email_corp else ""
        
        # =============================================================
        # [DADOS PROFISSIONAIS E FINANCEIROS] — PNAD 2024
        # =============================================================
        cargo = fake.job()
        
        # Salário — Distribuição Lognormal brasileira
        # Mediana real ~R$3.057 (PNAD 2024)
        # mean=7.9, sigma=0.75 → mediana ≈ e^7.9 ≈ R$2.700, média ≈ R$3.600
        # Distribuição: ~50% até R$3k, ~30% R$3k-8k, ~15% R$8k-15k, ~5% >R$15k
        salario = round(float(np.clip(np.random.lognormal(mean=7.9, sigma=0.75), 1412, 40000)), 2)
        
        # Setor — PIB brasileiro
        setor = np.random.choice(setores, p=pesos_setor)
        
        # Tamanho empresa — SEBRAE (97% micro/pequenas)
        tamanho_empresa = np.random.choice(tamanhos_empresa, p=pesos_tamanho)
        
        # Correlação: salário ajustado por tamanho da empresa
        # Empresas maiores pagam melhor
        if tamanho_empresa == "500+":
            salario = round(salario * np.random.uniform(1.5, 2.5), 2)
        elif tamanho_empresa == "201-500":
            salario = round(salario * np.random.uniform(1.2, 1.8), 2)
        elif tamanho_empresa == "51-200":
            salario = round(salario * np.random.uniform(1.0, 1.4), 2)
        # Micro empresas mantêm o salário base
        
        # Teto máximo realista
        salario = min(salario, 45000.0)
        
        # =============================================================
        # [DADOS GEOGRÁFICOS]
        # =============================================================
        endereco_sorteado = random.choice(cache_enderecos)
        
        # =============================================================
        # [DADOS DIGITAIS] — StatCounter Brasil 2024
        # =============================================================
        origem = np.random.choice(origens, p=pesos_origem)
        dispositivo = np.random.choice(dispositivos, p=pesos_dispositivo)
        
        # SO correlacionado com o dispositivo
        so_opcoes, so_pesos = so_por_dispositivo[dispositivo]
        sistema_operacional = np.random.choice(so_opcoes, p=so_pesos)
        
        # =============================================================
        # [COMPORTAMENTO DE ENGAJAMENTO] — Distribuição exponencial
        # Maioria visita poucas páginas / fica pouco tempo
        # =============================================================
        
        # Páginas visitadas — Exponencial (maioria 1-5, poucos >15)
        paginas_visitadas = int(np.clip(np.random.exponential(4), 1, 30))
        
        # Tempo no site — Lognormal (maioria <5min, cauda até 45min)
        tempo_site_min = round(float(np.clip(np.random.lognormal(1.0, 0.8), 0.3, 45.0)), 2)
        
        # E-mail — Taxas realistas de abertura (~25%) e clique (~3-5% do total)
        abriu_email = 1 if random.random() < 0.25 else 0
        # Só pode clicar se abriu — taxa de clique ~15-20% dos que abriram
        clicou_email = (1 if random.random() < 0.18 else 0) if abriu_email else 0
        
        # =============================================================
        # MONTAGEM DO LEAD
        # =============================================================
        lead = {
            # Pessoais
            "Nome_Completo": nome,
            "CPF": fake.cpf(),
            "RG": fake.rg(),
            "Data_Nascimento": data_nascimento,
            "Genero": genero,
            "Estado_Civil": estado_civil,
            
            # Contato
            "Email_Pessoal": email_pessoal,
            "Email_Corporativo": email_corporativo,
            "Telefone": fake.cellphone_number(),
            
            # Profissionais e Financeiros
            "Empresa": fake.company(),
            "CNPJ": fake.cnpj(),
            "Cargo": cargo,
            "Salario": salario,
            "Setor": setor,
            "Tamanho_Empresa": tamanho_empresa,
            
            # Geográficos
            "CEP": endereco_sorteado["CEP"],
            "Logradouro": endereco_sorteado["Logradouro"],
            "Numero": fake.building_number(),
            "Complemento": random.choice(complementos),
            "Bairro": endereco_sorteado["Bairro"],
            "Cidade": endereco_sorteado["Cidade"],
            "Estado": endereco_sorteado["Estado"],
            
            # Digitais
            "Origem": origem,
            "Dispositivo": dispositivo,
            "Sistema_Operacional": sistema_operacional,
            "IP": fake.ipv4(),
            
            # Comportamentais
            "Paginas_Visitadas": paginas_visitadas,
            "Tempo_Site_Min": tempo_site_min,
            "Abriu_Email": abriu_email,
            "Clicou_Email": clicou_email
        }
        
        # Motor de probabilidade: calcula score baseado em features reais
        score_deterministico = calcular_score_deterministico(lead, perfil)
        score_final = score_deterministico * PESO_DETERMINISTICO + random.random() * PESO_ALEATORIO
        lead["Score_Deterministico"] = round(score_deterministico, 6)
        lead["Score_Final"] = round(score_final, 6)
        lead["Status_Venda"] = 0
        
        leads.append(lead)
        
    leads.sort(key=lambda l: l["Score_Final"], reverse=True)
    if num_clientes is not None:
        alvo_conversoes = max(0, min(int(num_clientes), num_leads))
        msg_alvo = f"exatamente {alvo_conversoes} conversões"
    else:
        taxa_ajustada = max(0.0, min(float(taxa_conversao), 1.0))
        alvo_conversoes = int(round(num_leads * taxa_ajustada))
        msg_alvo = f"{taxa_ajustada * 100:.2f}% de conversão ({alvo_conversoes} vendas)"

    for idx, lead in enumerate(leads):
        lead["Status_Venda"] = 1 if idx < alvo_conversoes else 0

    print(f"  -> Vendas calibradas por score para {msg_alvo}.")
    random.shuffle(leads)
        
    print("[+] Fase B concluída: Leads gerados com distribuições reais do Brasil (IBGE/PNAD 2024).\n")
    return leads

# =====================================================================
# FASE D: PROCESSAMENTO E EXPORTAÇÃO
# =====================================================================
def processar_e_exportar_dados(leads):
    print("[*] Iniciando Fase D: Exportação / Validação de Cache...")
    
    df_completo = pd.DataFrame(leads)
    
    # Verifica se os arquivos finais já existem para evitar gravação redundante
    if os.path.exists(ARQUIVO_LEADS) and os.path.exists(ARQUIVO_CLIENTES):
        print(f"  -> Os arquivos '{ARQUIVO_LEADS}' e '{ARQUIVO_CLIENTES}' já existem e não serão recriados.")
        df_vendas = pd.read_parquet(ARQUIVO_CLIENTES)
        return df_completo, df_vendas

    # Se não existirem, filtra a tabela de clientes e salva ambos
    df_vendas = df_completo[df_completo["Status_Venda"] == 1]
    
    df_completo.to_parquet(ARQUIVO_LEADS, index=False, engine='pyarrow')
    df_vendas.to_parquet(ARQUIVO_CLIENTES, index=False, engine='pyarrow')
    
    print(f"  -> Arquivo criado: {ARQUIVO_LEADS}")
    print(f"  -> Arquivo criado: {ARQUIVO_CLIENTES}\n")
    
    return df_completo, df_vendas

# =====================================================================
# EXECUÇÃO PRINCIPAL DO SCRIPT
# =====================================================================
def main():
    tempo_inicio = time.time()
    
    print("\n" + "="*70)
    print(" G E R A D O R   D E   D A D O S   S I N T É T I C O S   (M L)")
    print("="*70 + "\n")

    modo_negocio = os.getenv("LEADLAB_MODO_NEGOCIO", "b2b").strip().lower()
    escopo_geografico = os.getenv("LEADLAB_ESCOPO_GEOGRAFICO", "regional").strip().lower()
    
    try:
        perfil = obter_perfil_mercado(modo_negocio=modo_negocio, escopo_geografico=escopo_geografico)

        # Executa FASE A
        cache = gerar_cache_enderecos(
            quantidade=QTD_CEPS_VALIDOS,
            estados_alvo=perfil["estados_alvo"],
        )

        # Executa FASE B
        dados_gerados = gerar_dados_sinteticos(
            cache,
            num_leads=QTD_LEADS_TOTAL,
            num_clientes=QTD_CLIENTES_ALVO,
            modo_negocio=modo_negocio,
            escopo_geografico=escopo_geografico,
        )

        # Executa FASE D
        df_all, df_conversions = processar_e_exportar_dados(dados_gerados)
    except Exception as e:
        print(f"[!] Falha durante a geração de dados: {e}")
        return
    
    # Cálculos das Métricas Finais
    tempo_fim = time.time()
    tempo_total = tempo_fim - tempo_inicio
    
    total_leads = len(df_all)
    total_colunas = len(df_all.columns)
    total_vendas = len(df_conversions)
    taxa_conversao = (total_vendas / total_leads) * 100 if total_leads > 0 else 0
    
    print("="*70)
    print(" M É T R I C A S   F I N A I S   D E   E X E C U Ç Ã O")
    print("="*70)
    print(f" Tempo Total de Execução : {tempo_total:.2f} segundos")
    print(f" Total de Leads (Geral)  : {total_leads}")
    print(f" Total de Colunas        : {total_colunas}")
    print(f" Vendas (Clientes)       : {total_vendas}")
    print(f" Taxa de Conversão Final : {taxa_conversao:.2f}%")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
