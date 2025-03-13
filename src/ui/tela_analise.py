import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from fpdf import FPDF
import torch
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os

# Definição do modelo (ajuste de acordo com o seu modelo real)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 64 * 64, 2)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 64)
        x = self.fc1(x)
        return x

# Função de pré-processamento
def preprocess_image(image_path):
    image = np.load(image_path)  # Carregar imagem .npy
    image = np.squeeze(image)    # Remover dimensões desnecessárias (caso a imagem tenha um canal extra)
    image = Image.fromarray(image.astype('uint8'))  # Converter para imagem PIL
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def calcular_angulo_cobb():
    # Simulação do cálculo do Ângulo de Cobb (deve ser ajustado conforme a análise)
    return 45  # Substitua com o cálculo real do Ângulo de Cobb

def exportar_relatorio_pdf(resultado, angulo_cobb, caminho_imagem, imagem_processada):
    # Pasta onde os relatórios serão salvos (agora será 'exportados' na raiz)
    pasta_exportados = "exportados/"
    
    # Criação da pasta se ela não existir
    if not os.path.exists(pasta_exportados):
        os.makedirs(pasta_exportados)
    
    # Encontrar o próximo número sequencial para o nome do arquivo
    relatorios_existentes = [f for f in os.listdir(pasta_exportados) if f.startswith("relatorio_")]
    numeros = [int(f.split('_')[1].split('.')[0]) for f in relatorios_existentes if f.split('_')[1].isdigit()]
    numero_sequencial = max(numeros, default=0) + 1  # Incrementa o último número encontrado
    
    pdf_nome = f"relatorio_{numero_sequencial}.pdf"

    # Criação do PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Relatório da Análise de Escoliose", ln=True, align="C")
    pdf.ln(10)
    
    pdf.cell(200, 10, txt=f"Resultado: {resultado}", ln=True)
    pdf.cell(200, 10, txt=f"Ângulo de Cobb: {angulo_cobb}°", ln=True)
    pdf.cell(200, 10, txt=f"Imagem analisada: {caminho_imagem}", ln=True)
    
    # Ajustar a qualidade da imagem para o PDF
    imagem_pdf_path = os.path.join(pasta_exportados, f"imagem_{numero_sequencial}.jpg")
    
    # Salvar a imagem processada com a qualidade adequada para o PDF
    imagem_processada = imagem_processada.convert("RGB")  # Converte para RGB se necessário
    imagem_processada.save(imagem_pdf_path, "JPEG", quality=95)  # Salvando com qualidade alta
    
    # Inserir imagem no PDF
    pdf.image(imagem_pdf_path, x=10, y=50, w=180)  # Ajustar a posição e tamanho conforme necessário
    
    # Salvar PDF na pasta 'exportados'
    caminho_pdf = os.path.join(pasta_exportados, pdf_nome)  # Salva o PDF na pasta 'exportados/'
    pdf.output(caminho_pdf)
    
    return caminho_pdf

class TelaAnalise(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load('modelo_treinado.pth'))
        self.model.eval()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.label_imagem = QLabel("Nenhuma imagem selecionada")
        self.layout.addWidget(self.label_imagem)
        
        self.btn_selecionar = QPushButton("Selecionar Imagem")
        self.btn_selecionar.clicked.connect(self.selecionar_imagem)
        self.layout.addWidget(self.btn_selecionar)

        self.btn_analisar = QPushButton("Analisar Imagem")
        self.btn_analisar.clicked.connect(self.analisar_imagem)
        self.layout.addWidget(self.btn_analisar)

        self.label_resultado = QLabel("Resultado: -")
        self.layout.addWidget(self.label_resultado)

        self.setLayout(self.layout)

    def selecionar_imagem(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Imagem", "", "Arquivos NPY (*.npy)")
        if file_path:
            self.image_path = file_path
            self.label_imagem.setText(f"Imagem: {file_path}")

    def analisar_imagem(self):
        if hasattr(self, 'image_path'):
            input_tensor = preprocess_image(self.image_path)
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            
            resultado = "Escoliose" if predicted.item() == 1 else "Sem Escoliose"
            angulo_cobb = calcular_angulo_cobb()
            
            # Exibir o resultado na tela
            self.label_resultado.setText(f"Resultado: {resultado} | Ângulo de Cobb: {angulo_cobb}°")
            
            # Carregar a imagem original com numpy e PIL
            imagem = np.load(self.image_path)  # Carregar a imagem .npy
            imagem = np.squeeze(imagem)  # Remover dimensões extras
            imagem = Image.fromarray(imagem.astype('uint8'))  # Converter para PIL
            draw = ImageDraw.Draw(imagem)
            font = ImageFont.load_default()  # Fonte padrão (ajuste se necessário)
            texto = f"Resultado: {resultado} | Ângulo de Cobb: {angulo_cobb}°"
            
            # Desenhar o texto na imagem
            draw.text((10, 10), texto, font=font, fill="white")
            
            # Não vamos exibir a imagem na interface gráfica
            # Converter para formato Qt para exibir na interface gráfica
            # imagem_qt = self.pil_to_qpixmap(imagem)
            # self.label_imagem.setPixmap(imagem_qt)
            
            # Exportar o relatório em PDF com um nome único sequencial
            caminho_pdf = exportar_relatorio_pdf(resultado, angulo_cobb, self.image_path, imagem)
            self.label_resultado.setText(f"Relatório exportado: {caminho_pdf}")
        else:
            self.label_resultado.setText("Nenhuma imagem selecionada!")

    def pil_to_qpixmap(self, pil_image):
        """Converter uma imagem PIL para QPixmap"""
        imagem_bytes = pil_image.tobytes()
        qim = QImage(imagem_bytes, pil_image.width, pil_image.height, pil_image.width * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qim)
