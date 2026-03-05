"""
Conversor de dumps SQL de CEP para parquet otimizado.

Etapas:
1) extrair linhas de INSERT dos arquivos SQL;
2) montar dataframes de cidade e logradouro;
3) realizar join por id_cidade;
4) exportar parquet final consumido pelo gerador de dados.
"""
import pandas as pd
import csv
import os

def ler_dump_sql(caminho_arquivo, colunas):
    """ETAPA 1 - Extrair tuplas de INSERT e retornar DataFrame."""
    print(f"[*] Lendo e extraindo dados de '{caminho_arquivo}'...")
    linhas_extraidas = []
    
    if not os.path.exists(caminho_arquivo):
        print(f"[!] ERRO: O arquivo {caminho_arquivo} não foi encontrado na pasta.")
        return pd.DataFrame(columns=colunas)

    # Lemos o arquivo ignorando possíveis erros de codificação de caracteres
    with open(caminho_arquivo, 'r', encoding='utf-8', errors='replace') as file:
        for linha in file:
            linha = linha.strip()
            
            # Pega apenas as linhas que são as "tuplas" de dados do INSERT INTO
            if linha.startswith("(") and (linha.endswith("),") or linha.endswith(");")):
                
                # Remove os parênteses das pontas: ( 'dado1', 'dado2' ) -> 'dado1', 'dado2'
                linha_limpa = linha[1:-2] if linha.endswith("),") else linha[1:-2]
                
                # O PULO DO GATO: Usa o leitor de CSV para interpretar as aspas simples 
                # do SQL corretamente, inclusive ignorando vírgulas dentro do texto
                leitor = csv.reader([linha_limpa], quotechar="'", skipinitialspace=True)
                for row in leitor:
                    linhas_extraidas.append(row)
                    
    df = pd.DataFrame(linhas_extraidas, columns=colunas)
    print(f"  -> {len(df)} registros extraídos.")
    return df

def converter_para_parquet():
    """ETAPA 2 a 4 - Carregar dumps, cruzar dados e exportar parquet."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ceps_dir = os.path.join(base_dir, 'ceps')
    
    # 1. Definir as colunas baseadas no CREATE TABLE do cidade.sql
    cols_cidade = ['id_cidade', 'nome_cidade', 'uf', 'codigo_ibge', 'ddd']
    df_cidade = ler_dump_sql(os.path.join(ceps_dir, 'cidade.sql'), cols_cidade)
    
    # 2. Definir as colunas baseadas no CREATE TABLE do logradouro.sql
    cols_logradouro = [
        'CEP', 'id_logradouro', 'tipo', 'descricao', 'id_cidade', 
        'UF', 'complemento', 'descricao_sem_numero', 'descricao_cidade', 
        'codigo_cidade_ibge', 'descricao_bairro'
    ]
    df_logradouro = ler_dump_sql(os.path.join(ceps_dir, 'logradouro.sql'), cols_logradouro)
    
    if df_logradouro.empty or df_cidade.empty:
        print("[!] Abortando: Um dos arquivos base está faltando ou vazio.")
        return

    print("\n[*] Cruzando dados de Ruas e Cidades (JOIN)...")
    # Garante que os IDs sejam lidos como texto para o cruzamento ser perfeito
    df_cidade['id_cidade'] = df_cidade['id_cidade'].astype(str)
    df_logradouro['id_cidade'] = df_logradouro['id_cidade'].astype(str)
    
    # Junta as duas tabelas
    df_completo = pd.merge(df_logradouro, df_cidade, on='id_cidade', how='left')

    print("[*] Limpando e Estruturando a Base Final...")
    # Monta a tabela final exatamente com as 5 colunas que o seu script de ML precisa
    df_final = pd.DataFrame({
        'CEP': df_completo['CEP'].str.replace('-', ''), # Garante CEP limpo
        'Logradouro': df_completo['descricao'],         # Pega o nome completo da rua
        'Bairro': df_completo['descricao_bairro'],
        # Usa o nome da cidade da tabela 'cidade', se falhar, pega da tabela 'logradouro'
        'Cidade': df_completo['nome_cidade'].fillna(df_completo['descricao_cidade']), 
        'Estado': df_completo['UF']
    })
    
    # Remove qualquer linha corrompida sem CEP
    df_final = df_final.dropna(subset=['CEP'])

    print("[*] Convertendo para formato Parquet de alta compressão...")
    arquivo_saida = os.path.join(ceps_dir, 'base_ceps_otimizada.parquet')
    
    # Salva o arquivo final
    df_final.to_parquet(arquivo_saida, engine='pyarrow', index=False)
    
    tamanho = os.path.getsize(arquivo_saida) / (1024 * 1024)
    print(f"\n[+] MÁGICA CONCLUÍDA! Arquivo '{arquivo_saida}' gerado com sucesso ({tamanho:.2f} MB).")
    print("Sua base de dados de endereços oficial agora está ultraleve!")

if __name__ == "__main__":
    converter_para_parquet()
