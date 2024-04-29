import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as Data
import numpy as np

# Define the Autoencoder class inheriting from nn.Module
class AE(nn.Module):
    def __init__(self, dim, emb_dim=128):
        super(AE, self).__init__()
        self.dim = dim  # input dimension
        # First layer of the encoder part
        self.fc1 = nn.Linear(dim, 512)  # Maps from input dimension to 512
        # Second layer of the encoder part, outputs the embedding
        self.fc2 = nn.Linear(512, emb_dim)  # Reduces dimensionality to emb_dim
        # First layer of the decoder part
        self.fc3 = nn.Linear(emb_dim, 512)  # Maps from embedding dimension back to 512
        # Final layer of the decoder part, outputs the reconstruction
        self.fc4 = nn.Linear(512, dim)  # Maps back to the original input dimension

    def encode(self, x):
        # Encoding function applying ReLU activation
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def decode(self, z):
        # Decoding function, reconstructs the input
        h3 = F.relu(self.fc3(z))
        return torch.relu(self.fc4(h3))

    def forward(self, x):
        # Forward pass through the network
        z = self.encode(x.view(-1, self.dim))  # Encodes the input
        return self.decode(z), z  # Returns the reconstruction and the embedding

# Function to perform dimensionality reduction using an autoencoder
def reduction_AE(gene_cell, device):
    # Convert gene data to tensor and transfer to device (GPU or CPU)
    gene = torch.tensor(gene_cell, dtype=torch.float32).to(device)
    # Determine batch size based on the size of data
    if gene_cell.shape[0] < 5000:
        ba = gene_cell.shape[0]  # Use full data if small enough
    else:
        ba = 5000  # Use maximum batch size of 5000
    # Train the autoencoder for gene data
    gene_embed = train_AE(gene, ba, device)

    # Process for cell data, similar to above but transposed
    if gene_cell.shape[1] < 5000:
        ba = gene_cell.shape[1]
    else:
        ba = 5000
    cell = torch.tensor(np.transpose(gene_cell), dtype=torch.float32).to(device)
    cell_embed = train_AE(cell, ba, device)

    return gene_embed, cell_embed

# Function to train the autoencoder
def train_AE(feature, ba, device, alpha=0.5, is_init=False):
    model = AE(dim=feature.shape[1]).to(device)  # Instantiate the AE model
    model.train()  # Set the model to training mode

    loader = Data.DataLoader(feature, batch_size=ba)  # Create data loader
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Define the optimizer

    loss_func = nn.MSELoss()  # Mean Squared Error loss for reconstruction
    EPOCH_AE = 2000  # Number of epochs to train
    for epoch in range(EPOCH_AE):
        embeddings = []  # List to store embeddings from each batch
        for _, batch_x in enumerate(loader):
            decoded, encoded = model(batch_x)  # Perform a forward pass
            loss = loss_func(batch_x, decoded)  # Compute the loss

            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagate the error
            optimizer.step()  # Update weights
            embeddings.append(encoded)  # Collect embeddings

        # Output training progress
        if epoch % 100 == 0:  # Print loss every 100 epochs
            print('Epoch :', epoch, '|', 'train_loss:%.12f' % loss.data)

    return torch.cat(embeddings)  # Concatenate all embeddings into a single tensor
