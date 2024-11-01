# use_cvae.py

import torch
import pandas as pd
from cva_model import CVAE
import joblib  # Para cargar label_encoder y otros objetos necesarios

# Cargar los objetos necesarios
label_encoder = joblib.load('label_encoder.pkl')
encoder = joblib.load('onehot_encoder.pkl')
X_for_cvae = pd.read_csv('X_for_cvae.csv')  # Si tienes los datos guardados
# Asegúrate de que X_for_cvae tenga las mismas columnas y orden que durante el entrenamiento

# Definir el modelo y cargar los pesos
input_dim = X_for_cvae.shape[1]
latent_dim = 50
num_classes = len(label_encoder.classes_)

cvae = CVAE(input_dim=input_dim, latent_dim=latent_dim, num_classes=num_classes)
cvae.load_state_dict(torch.load('cvae_model.pth'))
cvae.eval()

# Generar nuevas muestras
with torch.no_grad():
    num_new_samples = 100
    class_to_generate = 1  # Clase para la cual quieres generar datos
    y_new = torch.tensor([class_to_generate] * num_new_samples)
    z = torch.randn(num_new_samples, latent_dim)
    samples = cvae.decode(z, y_new)
    samples = samples.numpy()

# Postprocesamiento de las muestras
samples_df = pd.DataFrame(samples, columns=X_for_cvae.columns)

# Redondear variables binarias
for col in samples_df.columns:
    if set(X_for_cvae[col].unique()) == {0, 1}:
        samples_df[col] = (samples_df[col] > 0.5).astype(int)

# Agregar la clase generada
samples_df['clasificacion_final'] = label_encoder.inverse_transform([class_to_generate])

print("Nuevas muestras generadas:")
print(samples_df.head())
