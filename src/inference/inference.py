import os
import torch
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Definir a rede neural (modelo simples)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 1 canal de entrada (escala de cinza)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 64 * 64, 2)  # Ajustado para as dimensões das imagens de 128x128

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 64)  # Ajuste de dimensão
        x = self.fc1(x)
        return x

# Carregar o modelo treinado
model = SimpleCNN()
model.load_state_dict(torch.load('modelo_treinado.pth'))
model.eval()  # Coloca o modelo em modo de avaliação

# Função de pré-processamento
def preprocess_image(image_path):
    # Carregar a imagem e converter para escala de cinza
    image = np.load(image_path)  # Carregar o arquivo .jpg.npy
    image = Image.fromarray(image.astype('uint8'))  # Convertendo para uma imagem PIL

    # Transformação para redimensionar e converter para tensor
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Redimensiona para 128x128
        transforms.Grayscale(num_output_channels=1),  # Garantir 1 canal
        transforms.ToTensor(),  # Convertendo para tensor
    ])
    
    image = transform(image)
    image = image.unsqueeze(0)  # Adiciona uma dimensão para o batch (1, 1, 128, 128)
    return image

# Caminho da pasta contendo as imagens
folder_path = 'C:/Users/Victo/OneDrive/Documentos/EscolioseAI/data/sem_escoliose'

# Percorrer todas as imagens na pasta
for file_name in os.listdir(folder_path):
    if file_name.endswith('.jpg.npy'):  # Verifica se o arquivo tem a extensão esperada
        image_path = os.path.join(folder_path, file_name)

        # Pré-processar a imagem
        input_tensor = preprocess_image(image_path)

        # Realizar inferência
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

        # Exibir o resultado para cada imagem
        print(f'Imagem: {file_name} -> Predição: {"Escoliose" if predicted.item() == 1 else "Sem Escoliose"}')
