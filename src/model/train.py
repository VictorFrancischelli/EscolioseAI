import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
from src.model.cnn_classificacao import CNNModel

# Dataset personalizado
class ScoliosisDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.label_map = {"sem_escoliose": 0, "leve": 1, "moderada": 2, "grave": 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.label_map[self.data.iloc[idx, 1]]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label

# Configuração do treinamento
def train_model():
    # Caminho do dataset
    csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset.csv'))
    dataset = ScoliosisDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Modelo
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Treinamento
    num_epochs = 20
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

    # Salvar o modelo
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'modelo_treinado.pth'))
    torch.save(model.state_dict(), output_path)
    print(f"Modelo treinado e salvo em {output_path}")

if __name__ == "__main__":
    train_model()