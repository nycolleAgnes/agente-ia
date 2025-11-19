# scripts/predict.py — Carrega o modelo salvo e faz previsões

import os
import pandas as pd
import joblib

# --- Detecta ambiente: Colab ou local ---
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    BASE_DIR = "/content/drive/MyDrive/agente-ia"
    print(" Executando no Google Colab.")
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(" Executando localmente (VS Code).")

# 1. Definir caminhos seguros

model_dir = os.path.join(BASE_DIR, "model")
modelo_path = os.path.join(model_dir, "modelo_logistico.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

# 2. Carregar modelo e scaler
print(f" Carregando modelo de: {modelo_path}")
modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)
print(" Modelo e scaler carregados com sucesso!\n")

# 3. Criar novos dados de exemplo (ou carregar de CSV)
#  Os nomes das colunas devem ser os mesmos da base original (sem 'inadimplente')
novos_clientes = pd.DataFrame({
    "client_id": [1001, 1002, 1003],
    "idade": [25, 45, 60],
    "renda_mensal": [2500, 8000, 12000],
    "tempo_emprego_anos": [1.2, 5.4, 10.1],
    "qtd_contas": [1, 3, 2],
    "score_credito": [580, 700, 820],
    "historico_atrasos": [2, 0, 1],
    "valor_emprestimo": [5000, 40000, 100000],
    "parcelas": [12, 36, 48]
})

# 4. Escalar os dados (sem o client_id)
novos_clientes_scaled = scaler.transform(
    novos_clientes.drop("client_id", axis=1)
)

# 5. Fazer previsões
predicoes = modelo.predict(novos_clientes_scaled)
probabilidades = modelo.predict_proba(novos_clientes_scaled)[:, 1]

# 6. Exibir resultados
resultados = novos_clientes.copy()
resultados["Prob_Inadimplente"] = probabilidades
resultados["Predição"] = ["Inadimplente" if p == 1 else "Adimplente" for p in predicoes]

print("=== Previsões para novos clientes ===")
print(resultados.to_string(index=False))

# (Opcional) Salvar resultados em CSV
saida_csv = os.path.join(BASE_DIR, "data", "previsoes_clientes.csv")
resultados.to_csv(saida_csv, index=False)
print(f"\n Resultados salvos em: {saida_csv}")

