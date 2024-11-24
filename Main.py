from data_processing import load_and_process_data, balance_data
from model_training import train_gb_model, save_model
from cvae import CVAE, train_cvae, save_cvae_model
from utils import generate_new_samples
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score


# if __name__ == "__main__":
#     # Carga y procesamiento de datos
#     X, y, label_encoder, encoder = load_and_process_data('bd_denge1.csv', 'bd_dengue.csv')
    
#     # Equilibrio de clases
#     X_balanced, y_balanced = balance_data(X, y)
    
#     # Convertir X_balanced y y_balanced a tensores
#     X_tensor = torch.tensor(X_balanced.values, dtype=torch.float32)
#     y_tensor = torch.tensor(y_balanced.values, dtype=torch.long)
    
#     # Crear un DataLoader para CVAE
#     dataset = TensorDataset(X_tensor, y_tensor)
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
#     # Entrenamiento de modelos
#     gb_model = train_gb_model(X_balanced, y_balanced)
#     save_model(gb_model, label_encoder, encoder)
    
#     # Evaluar el modelo
#     y_pred = gb_model.predict(X_balanced)
#     accuracy = accuracy_score(y_balanced, y_pred)
#     report = classification_report(y_balanced, y_pred, target_names=label_encoder.classes_)
#     print("========== Modelo de Gradient Boosting ==========")
#     print(f"Accuracy: {accuracy:.4f}")
#     print("Classification Report:\n", report)
    
#     # Visualizar la matriz de confusión
#     cm = confusion_matrix(y_balanced, y_pred)
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.show()
    
#     # Entrenar y guardar CVAE
#     cvae = CVAE(input_dim=X.shape[1], latent_dim=50, num_classes=len(label_encoder.classes_))
#     print(X.shape[1], len(label_encoder.classes_), "data")

#     # Asegúrate de que train_cvae reciba el DataLoader y el valor numérico de num_epochs
#     train_cvae(cvae, dataloader, num_epochs=20)  # Aquí pasamos un entero (20) como num_epochs
#     save_cvae_model(cvae)

#     # Generar muestras nuevas
#     generated_samples = generate_new_samples(cvae)
#     print(generated_samples)

