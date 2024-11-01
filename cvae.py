import torch
import torch.nn as nn
import torch.optim as optim

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
    

def train_cvae(cvae, dataloader, num_epochs=20):
    optimizer = optim.Adam(cvae.parameters(), lr=1e-3)
    loss_function = nn.BCELoss()

    for epoch in range(num_epochs):
        train_loss = 0
        for batch in dataloader:
            data, labels = batch
            optimizer.zero_grad()
            recon_batch, mu, logvar = cvae(data, labels)
            loss = loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')
    return cvae

def save_cvae_model(cvae, model_path='cvae_model.pth'):
    torch.save(cvae.state_dict(), model_path)
