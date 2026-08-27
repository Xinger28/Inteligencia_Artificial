import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# =====================================================================
# 1. VERIFICAR Y CONFIGURAR EL DISPOSITIVO (GPU vs CPU)
# =====================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--> Usando dispositivo: {device}")

if device.type == 'cuda':
    print(f"--> Tarjeta detectada: {torch.cuda.get_device_name(0)}\n")
else:
    print("--> ADVERTENCIA: CUDA no está disponible, se usará la CPU.\n")

# =====================================================================
# 2. GENERAR Y PREPARAR EL DATASET (Machine Learning Tradicional)
# =====================================================================
# Crear dataset sintético de 5,000 muestras con 10 características (features)
X_raw, y_raw = make_classification(
    n_samples=5000, 
    n_features=10, 
    n_classes=2, 
    random_state=42
)

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# Escalado de características (buena práctica para ML)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================================
# 3. MOVER NATIVAMENTE LOS DATOS A LA GPU (Tensores de PyTorch)
# =====================================================================
# Convertir arreglos NumPy a Tensores de PyTorch y enviarlos a la GPU con .to(device)
X_train_gpu = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_gpu = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

X_test_gpu = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_gpu = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# =====================================================================
# 4. DEFINIR EL MODELO DE REGRESIÓN LOGÍSTICA
# =====================================================================
class LogisticRegressionGPU(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionGPU, self).__init__()
        # Una capa lineal seguida de una función Sigmoide representa la Regresión Logística
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Instanciar el modelo y MOVERLO a la GPU
model = LogisticRegressionGPU(input_dim=10).to(device)

# Definir la función de pérdida (Binary Cross Entropy) y el optimizador (Adam)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# =====================================================================
# 5. ENTRENAMIENTO EN LA TARJETA GRÁFICA
# =====================================================================
epochs = 100
print("--- Iniciando entrenamiento en GPU ---")

for epoch in range(epochs):
    model.train()
    
    # Pasada hacia adelante (Forward pass) en GPU
    outputs = model(X_train_gpu)
    loss = criterion(outputs, y_train_gpu)
    
    # Pasada hacia atrás (Backward pass) y optimización
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Época [{epoch+1}/{epochs}] - Pérdida (Loss): {loss.item():.4f}")

# =====================================================================
# 6. EVALUACIÓN Y PRECISIÓN DEL MODELO
# =====================================================================
model.eval()
with torch.no_grad():
    predictions = model(X_test_gpu)
    # Convertir probabilidades a clases binarias (0 o 1)
    predicted_classes = (predictions >= 0.5).float()
    
    # Calcular la precisión directamente en la GPU
    accuracy = (predicted_classes.eq(y_test_gpu).sum() / y_test_gpu.shape[0]).item()

print("\n--- Resultados ---")
print(f"Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%")