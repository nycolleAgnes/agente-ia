#  01_gera_dados.ipynb

# Compatível com Google Colab e VS Code

import numpy as np
import pandas as pd
import os

# Detectar ambiente (Colab ou local)
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Definir caminho base 
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    base_path = "/content/drive/MyDrive/agente-ia/data"
else:
    base_path = os.path.join(os.getcwd(), "../data")  # caminho relativo local

# Criar pasta se não existir 
os.makedirs(base_path, exist_ok=True)
print(f" Pasta de dados verificada/criada em: {base_path}")

# Configurações iniciais 
pd.set_option("display.float_format", "{:,.2f}".format)
np.random.seed(42)

# Geração dos dados 
n = 5000
print(f" Serão gerados {n} registros de clientes...")

idade = np.random.randint(18, 75, size=n)
renda_mensal = np.round(np.random.lognormal(mean=8, sigma=0.8, size=n), 2)
tempo_emprego = np.round(np.random.exponential(scale=3, size=n), 1)
qtd_contas = np.random.poisson(lam=2, size=n)
score_credito = np.clip(np.random.normal(loc=600, scale=100, size=n), 300, 850)
historico_atrasos = np.random.poisson(lam=0.5, size=n)
valor_emprestimo = np.round(np.random.uniform(500, 50000, size=n), 2)
parcelas = np.random.randint(3, 60, size=n)

# Simulação de risco e inadimplência
risk = (
    (score_credito < 550).astype(int) * 1.5 +
    (historico_atrasos > 0).astype(int) * 1.2 +
    (valor_emprestimo / (renda_mensal + 1) > 10).astype(int) * 1.3 +
    (tempo_emprego < 1).astype(int) * 1.0
)

prob = 1 / (1 + np.exp(-(-2 + 0.5 * risk)))
inadimplente = (np.random.rand(n) < prob).astype(int)

# Montagem do DataFrame
df = pd.DataFrame({
    "client_id": np.arange(1, n + 1),
    "idade": idade,
    "renda_mensal": renda_mensal,
    "tempo_emprego_anos": tempo_emprego,
    "qtd_contas": qtd_contas,
    "score_credito": score_credito,
    "historico_atrasos": historico_atrasos,
    "valor_emprestimo": valor_emprestimo,
    "parcelas": parcelas,
    "inadimplente": inadimplente
})

# Salvando o dataset
output_path = os.path.join(base_path, "clientes_sinteticos.csv")
df.to_csv(output_path, index=False)

print(f" Dataset salvo em: {output_path}")
print(f" Total de linhas: {len(df)}")
