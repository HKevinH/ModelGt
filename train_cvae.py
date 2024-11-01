# train_cvae.py

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from cvae_model import CVAE, loss_function
import joblib  # Si necesitas cargar otros objetos
import pandas as pd

# Cargar y preprocesar los datos (puedes reutilizar tu código existente)
# Asegúrate de cargar X_for_cvae y y_for_cvae

# Convertir los datos a tensores de PyTorch
X_tensor = torch.tensor(X_for_cvae.values, dtype=torch.float32)
y_tensor = torch.tensor(y_for_cvae.values, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Definir el modelo
input_dim = X_for_cvae.shape[1]
latent_dim = 50
num_classes = len(label_encoder.classes_)  # Asegúrate de tener label_encoder cargado

cvae = CVAE(input_dim=input_dim, latent_dim=latent_dim, num_classes=num_classes)
optimizer = optim.Adam(cvae.parameters(), lr=1e-3)

# Entrenar el modelo
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

# Guardar el modelo entrenado
torch.save(cvae.state_dict(), 'cvae_model.pth')
