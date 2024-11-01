import torch

def generate_new_samples(cvae, num_samples=100, class_label=1, latent_dim=50):
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        y_label = torch.tensor([class_label] * num_samples)
        samples = cvae.decode(z, y_label)
        return samples.numpy()
