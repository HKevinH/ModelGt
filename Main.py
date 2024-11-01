# ============================
# Importación de Librerías
# ============================

# Librerías estándar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento y Modelado
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Para guardar y cargar modelos
import joblib  # Añadido para exportar el modelo de scikit-learn

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Configuración adicional
import warnings
warnings.filterwarnings('ignore')

# ============================
# Carga y Preprocesamiento de Datos
# ============================

file_path1 = 'bd_denge1.csv'
file_path2 = 'bd_dengue.csv'
data_sivigila = pd.read_csv(file_path1, encoding='latin1', sep=';')
data_dengue = pd.read_csv(file_path2, encoding='latin1', sep=';')

all_columns = [
    'codigo_sspd', 'fecha_nototificacion', 'edad', 'nacionalidad', 'sexo', 'comuna',
    'tipo_seguridad_social', 'etnia', 'estrato', 'ciudad_residencia', 'dpto_residencia',
    'fecha_consulta', 'fecha_inicio_sintomas', 'tipo_caso', 'Hospitalizado',
    'fecha_de_hospitalizacion', 'deplazamiento_ultimos_15_dias', 'municipio_desplazamiento',
    'fiebre', 'cefalea', 'dolor_retro_ocular', 'malgias', 'artralgia', 'erupcion',
    'dolor_abdominal', 'vomito', 'diarrea', 'somnolencia', 'hipotension', 'hepatomegalia',
    'hem_mucosa', 'hipotermia', 'aum_hemato', 'caida_plaquetas', 'acumulacion_liquidos',
    'extravasacion', 'hemorragia', 'choque', 'daño_organo', 'clasificacion_final',
    'conducta', 'nombre_municipio_procedencia'
]

symptom_columns = [
    'fiebre', 'cefalea', 'dolor_retro_ocular', 'malgias', 'artralgia', 'erupcion',
    'dolor_abdominal', 'vomito', 'diarrea', 'somnolencia', 'hipotension', 'hepatomegalia',
    'hem_mucosa', 'hipotermia', 'aum_hemato', 'caida_plaquetas', 'acumulacion_liquidos',
    'extravasacion', 'hemorragia', 'choque', 'daño_organo'
]

location_columns = ['ciudad_residencia', 'dpto_residencia', 'nombre_municipio_procedencia']

target_column = 'clasificacion_final'

# Filtrar el dataset para incluir solo las columnas seleccionadas
data_filtered = data_sivigila[all_columns]

# Convertir todas las cadenas a minúsculas
data_filtered = data_filtered.applymap(lambda s: s.lower() if type(s) == str else s)

# Reemplazar variaciones de 'sí'/'si' y 'no' por valores binarios
data_filtered = data_filtered.replace({'sí': 1, 'si': 1, 'no': 0})

# Convertir las columnas de síntomas a numérico, convirtiendo errores a NaN
data_filtered[symptom_columns] = data_filtered[symptom_columns].apply(pd.to_numeric, errors='coerce')

# Eliminar filas con valores NaN en síntomas y en las columnas necesarias
data_filtered = data_filtered.dropna(subset=symptom_columns + [target_column] + location_columns)

# Codificar la variable objetivo
label_encoder = LabelEncoder()
data_filtered[target_column] = label_encoder.fit_transform(data_filtered[target_column])

# ============================
# Análisis de Tendencias por Municipio
# ============================

# Contar casos por municipio
cases_by_municipality = data_filtered['nombre_municipio_procedencia'].value_counts().reset_index()
cases_by_municipality.columns = ['Municipio', 'Casos']

# Mostrar los 10 municipios con más casos
top_municipalities = cases_by_municipality.head(10)
print("Top 10 municipios con más casos de dengue:")
print(top_municipalities)

# Graficar los casos por municipio
plt.figure(figsize=(12, 6))
sns.barplot(data=top_municipalities, x='Municipio', y='Casos')
plt.title('Top 10 municipios con más casos de dengue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================
# Preprocesamiento de Datos para el Modelo
# ============================

# Codificar variables categóricas (ubicación)
categorical_columns = location_columns + ['sexo', 'tipo_seguridad_social', 'etnia', 'estrato', 'nacionalidad']
data_filtered[categorical_columns] = data_filtered[categorical_columns].astype(str)

# Utilizar OneHotEncoder para variables categóricas
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_cat = encoder.fit_transform(data_filtered[categorical_columns])

encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_columns))

# Combinar datos numéricos y categóricos codificados
X = pd.concat([data_filtered[symptom_columns].reset_index(drop=True), encoded_cat_df], axis=1)
y = data_filtered[target_column]

# ============================
# Equilibrio de Clases
# ============================

# Combinar características y objetivo
data_combined = pd.concat([X, y.reset_index(drop=True)], axis=1)

# Identificar la clase mayoritaria y minoritaria
class_counts = data_combined[target_column].value_counts()
majority_class = class_counts.idxmax()
minority_class = class_counts.idxmin()

majority = data_combined[data_combined[target_column] == majority_class]
minority = data_combined[data_combined[target_column] == minority_class]

# Aumentar la clase minoritaria
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

# Combinar las clases para obtener un dataset equilibrado
data_balanced = pd.concat([majority, minority_upsampled])

# Separar características y objetivo después del balanceo
X_balanced = data_balanced.drop(columns=[target_column])
y_balanced = data_balanced[target_column]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# ============================
# Entrenamiento del Modelo
# ============================

# Entrenar el modelo de Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
gb_model.fit(X_train_bal, y_train_bal)

# Evaluar el modelo
y_pred = gb_model.predict(X_test_bal)
accuracy = accuracy_score(y_test_bal, y_pred)
report = classification_report(y_test_bal, y_pred, target_names=label_encoder.classes_)

print("========== Modelo de Gradient Boosting ==========")
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Guardar el modelo entrenado y los objetos necesarios
joblib.dump(gb_model, 'gb_model.pkl')  # Exportar el modelo de clasificación
joblib.dump(label_encoder, 'label_encoder.pkl')  # Exportar el codificador de etiquetas
joblib.dump(encoder, 'onehot_encoder.pkl')  # Exportar el OneHotEncoder

# ============================
# Implementación del Autoencoder Variacional Condicional (CVAE)
# ============================

# Preparar los datos para el CVAE
X_for_cvae = data_balanced.drop(columns=[target_column]).astype(float)
y_for_cvae = data_balanced[target_column].astype(int)

# Convertir los datos a tensores de PyTorch
X_tensor = torch.tensor(X_for_cvae.values, dtype=torch.float32)
y_tensor = torch.tensor(y_for_cvae.values, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Definir el modelo CVAE
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Capas del codificador
        self.fc1 = nn.Linear(input_dim + num_classes, 512)
        self.fc21 = nn.Linear(512, latent_dim)  # Media
        self.fc22 = nn.Linear(512, latent_dim)  # Log-varianza

        # Capas del decodificador
        self.fc3 = nn.Linear(latent_dim + num_classes, 512)
        self.fc4 = nn.Linear(512, input_dim)

    def encode(self, x, y):
        y_onehot = torch.zeros(x.size(0), self.num_classes)
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        x = torch.cat([x, y_onehot], dim=1)
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_onehot = torch.zeros(z.size(0), self.num_classes)
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        z = torch.cat([z, y_onehot], dim=1)
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, self.input_dim), y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# Definir la función de pérdida y el optimizador
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, x.shape[1]), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

cvae = CVAE(input_dim=X_for_cvae.shape[1], latent_dim=50, num_classes=len(label_encoder.classes_))
optimizer = optim.Adam(cvae.parameters(), lr=1e-3)

# Entrenar el CVAE
num_epochs = 20
cvae.train()
for epoch in range(num_epochs):
    train_loss = 0
    for batch in dataloader:
        data, labels = batch
        optimizer.zero_grad()
        recon_batch, mu, logvar = cvae(data, labels)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('Epoch: {} \t Loss: {:.6f}'.format(epoch+1, train_loss / len(dataloader.dataset)))

# Guardar el modelo CVAE
torch.save(cvae.state_dict(), 'cvae_model.pth')  # Exportar el modelo CVAE

# ============================
# Generar Nuevos Datos Condicionados
# ============================

# Generar nuevas muestras con el CVAE entrenado
cvae.eval()
with torch.no_grad():
    num_new_samples = 100  # Número de nuevas muestras a generar
    # Seleccionar la clase para la cual queremos generar datos (por ejemplo, 0 o 1)
    class_to_generate = 1  # Puedes cambiar esto para generar datos de otra clase
    y_new = torch.tensor([class_to_generate] * num_new_samples)
    z = torch.randn(num_new_samples, cvae.latent_dim)
    samples = cvae.decode(z, y_new)
    samples = samples.numpy()

# Postprocesamiento de los datos generados
samples_df = pd.DataFrame(samples, columns=X_for_cvae.columns)

# Redondear variables binarias (síntomas y variables categóricas codificadas)
for col in samples_df.columns:
    if set(X_for_cvae[col].unique()) == {0, 1}:
        samples_df[col] = (samples_df[col] > 0.5).astype(int)

# Agregar la clase generada
samples_df[target_column] = class_to_generate

print("========== Nuevas Muestras Generadas ==========")
print(samples_df.head())

# ============================
# Interfaz Interactiva Mejorada
# ============================

def predict_dengue(model, label_encoder, encoder, symptom_columns, categorical_columns):
    print("\nIngrese los síntomas y datos del paciente:")
    user_input = {}
    # Ingresar síntomas
    for symptom in symptom_columns:
        while True:
            value = input(f"¿El paciente presenta {symptom.replace('_', ' ')}? (sí/no): ").strip().lower()
            if value in ['sí', 'si', 'no']:
                user_input[symptom] = 1 if value in ['sí', 'si'] else 0
                break
            else:
                print("Entrada inválida. Por favor, ingrese 'sí' o 'no'.")
    # Ingresar datos categóricos
    for cat in categorical_columns:
        value = input(f"Ingrese {cat.replace('_', ' ')}: ").strip().lower()
        user_input[cat] = value
    # Convertir a DataFrame
    input_df = pd.DataFrame([user_input])
    # Codificar variables categóricas
    input_cat_encoded = encoder.transform(input_df[categorical_columns])
    input_cat_df = pd.DataFrame(input_cat_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    # Combinar datos
    input_final = pd.concat([input_df[symptom_columns].reset_index(drop=True), input_cat_df], axis=1)
    # Asegurar que las columnas estén en el mismo orden que el modelo
    input_final = input_final.reindex(columns=X.columns, fill_value=0)
    # Realizar la predicción
    prediction = model.predict(input_final)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]
    print(f"\nLa clasificación del paciente es: {prediction_label}")
    # Generar muestra similar con CVAE
    generate_similar_sample(cvae, input_final, prediction, label_encoder)

def generate_similar_sample(cvae, input_data, class_label, label_encoder):
    cvae.eval()
    with torch.no_grad():
        # Convertir los datos de entrada a tensor
        x_input = torch.tensor(input_data.values, dtype=torch.float32)
        # Obtener la representación latente
        mu, logvar = cvae.encode(x_input, torch.tensor([class_label]))
        z = cvae.reparameterize(mu, logvar)
        # Generar nueva muestra
        generated_sample = cvae.decode(z, torch.tensor([class_label]))
        generated_sample = generated_sample.numpy()
        generated_df = pd.DataFrame(generated_sample, columns=input_data.columns)
        # Postprocesamiento
        for col in generated_df.columns:
            if set(X_for_cvae[col].unique()) == {0, 1}:
                generated_df[col] = (generated_df[col] > 0.5).astype(int)
        print("\nDatos generados similares al paciente ingresado:")
        print(generated_df.head())

# ============================
# Uso Interactivo del Modelo
# ============================

if __name__ == "__main__":
    # Cargar los modelos y objetos guardados
    gb_model = joblib.load('gb_model.pkl')  # Cargar el modelo de clasificación
    label_encoder = joblib.load('label_encoder.pkl')  # Cargar el codificador de etiquetas
    encoder = joblib.load('onehot_encoder.pkl')  # Cargar el OneHotEncoder

    # Cargar el modelo CVAE
    cvae = CVAE(input_dim=X_for_cvae.shape[1], latent_dim=50, num_classes=len(label_encoder.classes_))
    cvae.load_state_dict(torch.load('cvae_model.pth'))
    cvae.eval()

    #while True:
    #    predict_dengue(gb_model, label_encoder, encoder, symptom_columns, categorical_columns)
        #another = input("\n¿Desea realizar otra predicción? (sí/no): ").strip().lower()
       # if another not in ['sí', 'si']:
      #      print("Gracias por utilizar el sistema de predicción de dengue.")
     #       break
