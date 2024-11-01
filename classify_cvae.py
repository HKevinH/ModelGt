# classify_cvae.py

import torch
import pandas as pd
from cva_model import CVAE
import joblib

# Cargar el modelo y objetos necesarios
label_encoder = joblib.load('label_encoder.pkl')
encoder = joblib.load('onehot_encoder.pkl')
X_for_cvae = pd.read_csv('X_for_cvae.csv')

input_dim = X_for_cvae.shape[1]
latent_dim = 50
num_classes = len(label_encoder.classes_)

cvae = CVAE(input_dim=input_dim, latent_dim=latent_dim, num_classes=num_classes)
cvae.load_state_dict(torch.load('cvae_model.pth'))
cvae.eval()

# Supongamos que tienes nuevos datos para clasificar
def preprocess_input(user_input):
    # Convertir a DataFrame
    input_df = pd.DataFrame([user_input])
    # Codificar variables categóricas
    input_cat_encoded = encoder.transform(input_df[categorical_columns])
    input_cat_df = pd.DataFrame(input_cat_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    # Combinar datos
    input_final = pd.concat([input_df[symptom_columns].reset_index(drop=True), input_cat_df], axis=1)
    # Asegurar que las columnas estén en el mismo orden que durante el entrenamiento
    input_final = input_final.reindex(columns=X_for_cvae.columns, fill_value=0)
    return input_final

# Obtener entrada del usuario o de un archivo
user_input = {
    'fiebre': 1,
    'cefalea': 0,
    # ... agregar todos los síntomas y variables categóricas necesarias
}

input_data = preprocess_input(user_input)
input_tensor = torch.tensor(input_data.values, dtype=torch.float32)

# Realizar inferencia
with torch.no_grad():
    # Puedes proporcionar la clase si la conoces, o probar con todas las clases
    y_dummy = torch.tensor([0])  # Clase dummy si no se conoce
    recon_data, mu, logvar = cvae(input_tensor, y_dummy)
    # Aquí podrías implementar lógica adicional para clasificar o interpretar los resultados

print("Resultado de la clasificación o inferencia:")
# Añade aquí cómo interpretar y mostrar los resultados
