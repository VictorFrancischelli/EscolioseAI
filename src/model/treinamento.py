import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Definir a classe para o dataset personalizado
class NumpyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Buscando as subpastas e arquivos .jpg.npy
        self.image_files = []
        self.labels = []
        for label in ['escoliose', 'sem_escoliose']:
            class_folder = os.path.join(root_dir, label)
            for file_name in os.listdir(class_folder):
                if file_name.endswith('.jpg.npy'):
                    self.image_files.append(os.path.join(class_folder, file_name))
                    self.labels.append(label)

        # Verificação para garantir que arquivos .jpg.npy foram encontrados
        if len(self.image_files) == 0:
            raise ValueError("Nenhum arquivo .jpg.npy encontrado nas pastas de classe especificadas.")
        print(f"Arquivos encontrados: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = np.load(img_name)  # Carregar o arquivo .jpg.npy
        label = self.labels[idx]

        # Converter a imagem para PIL antes de aplicar as transformações
        image = Image.fromarray(image.astype('uint8'))

        if self.transform:
            image = self.transform(image)

        # Garantir que o tipo de dado seja float32
        image = torch.tensor(image).float()  # Converter para float32

        # Converte o label para uma variável binária (0 ou 1)
        label = 0 if label == 'sem_escoliose' else 1

        return image, label


# Definir a rede neural (modelo simples)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Modificar para 1 canal de entrada, já que suas imagens são provavelmente em escala de cinza
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # Agora usa 1 canal de entrada, não 3
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 64 * 64, 2)  # Supondo que a imagem tenha 128x128 de tamanho

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 64)  # Ajuste de dimensão
        x = self.fc1(x)
        return x


# Função de treino
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


# Caminho para os dados
root_dir = 'C:/Users/Victo/OneDrive/Documentos/EscolioseAI/data'

# Definir transformações
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Ajuste para as imagens 128x128
    transforms.Grayscale(num_output_channels=1),  # Garantir que as imagens sejam 1 canal (escala de cinza)
    transforms.ToTensor(),
])

# Criar o dataset e o dataloader
train_dataset = NumpyDataset(root_dir=root_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Inicializar o modelo, critério de perda e otimizador
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento do modelo
train(model, train_loader, criterion, optimizer, num_epochs=10)

# Salvando o modelo treinado
torch.save(model.state_dict(), 'modelo_treinado.pth')
print('Modelo treinado salvo com sucesso!')
