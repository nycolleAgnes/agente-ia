# scripts/train_model.py — Treinamento e salvamento do modelo

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 1. Definir caminhos seguros

# Detecta ambiente: Colab ou local 
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/agente-ia"
    print("✅ Executando no Google Colab.")
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("✅ Executando localmente (VS Code).")

# Caminho dos dados e saída do modelo
data_path = os.path.join(BASE_DIR, "data", "clientes_sinteticos.csv")
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

# 2. Carregar os dados

print(f" Carregando dados de: {data_path}")
df = pd.read_csv(data_path)

# 3. Separar features e target

X = df.drop(["client_id", "inadimplente"], axis=1)
y = df["inadimplente"]


# 4. Dividir em treino e teste

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Padronizar os dados

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Treinar modelo

modelo = LogisticRegression(
    C=0.01,
    class_weight='balanced',
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
modelo.fit(X_train_scaled, y_train)

# 7. Avaliar desempenho

y_pred = modelo.predict(X_test_scaled)
print("\n=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred))

# 8. Salvar modelo e scaler

model_path = os.path.join(model_dir, "modelo_logistico.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

joblib.dump(modelo, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n Modelo salvo em: {model_path}")
print(f" Scaler salvo em: {scaler_path}")
print("\n Treinamento concluído com sucesso!")
