import os
import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from src.model.cnn_classificacao import CNNModel

# Carregar o modelo treinado
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'modelo_treinado.pth'))
model = CNNModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Função de pré-processamento para a CNN
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

# Função pra redimensionar a imagem pra exibição
def resize_for_display(img, max_height=800):
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

# Função pra calcular o ângulo de Cobb e desenhar as marcações
def detect_cobb_points(image_path, debug=True):
    # (mantive a função original)
    if not os.path.exists(image_path):
        print(f"Arquivo não encontrado: {image_path}")
        return None, None
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return None, None
    
    original_img = img.copy()
    img_smooth = cv2.GaussianBlur(original_img, (5, 5), 0)
    h, w = img_smooth.shape
    spine_centers = []
    for y in range(h):
        row = img_smooth[y, :]
        if np.sum(row) > 0:
            x_center = int(np.average(np.arange(w), weights=row))
            spine_centers.append((x_center, y))
    
    if len(spine_centers) < 5:
        print(f"Falha na detecção: {image_path}")
        return None, None
    
    spine_centers = np.array(spine_centers)
    x_centers, y_centers = spine_centers[:, 0], spine_centers[:, 1]
    coeffs = np.polyfit(y_centers, x_centers, 1)
    a, b = coeffs
    spine_line_x = a * y_centers + b
    
    window_size = 50
    top_slopes = []
    bottom_slopes = []
    for i in range(len(spine_centers) - window_size):
        segment = spine_centers[i:i + window_size]
        slope = np.polyfit(segment[:, 1], segment[:, 0], 1)[0]
        if segment[0, 1] < h // 2:
            top_slopes.append((slope, i))
        else:
            bottom_slopes.append((slope, i))
    
    if not top_slopes or not bottom_slopes:
        print(f"Falha nas inclinações: {image_path}")
        return None, None
    
    top_slope, top_start_idx = max(top_slopes, key=lambda x: abs(x[0]))
    bottom_slope, bottom_start_idx = max(bottom_slopes, key=lambda x: abs(x[0]))
    
    def calculate_angle(top_slope, bottom_slope):
        tan_theta = abs((bottom_slope - top_slope) / (1 + top_slope * bottom_slope))
        angle = np.degrees(np.arctan(tan_theta))
        spine_slope = coeffs[0]
        max_deviation = max(abs(top_slope - spine_slope), abs(bottom_slope - spine_slope))
        if max_deviation < 0.5:
            angle *= 0.5
        return angle
    
    angle = calculate_angle(top_slope, bottom_slope)
    
    debug_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    start_point = (int(spine_line_x[0]), y_centers[0])
    end_point = (int(spine_line_x[-1]), y_centers[-1])
    cv2.line(debug_img, start_point, end_point, (0, 0, 255), 2)
    line_length = 200
    thickness = 4
    top_x_start = int(spine_line_x[0] - line_length // 2)
    top_x_end = int(spine_line_x[0] + line_length // 2)
    cv2.line(debug_img, (top_x_start, y_centers[0]), (top_x_end, y_centers[0]), (0, 0, 255), thickness)
    bottom_x_start = int(spine_line_x[-1] - line_length // 2)
    bottom_x_end = int(spine_line_x[-1] + line_length // 2)
    cv2.line(debug_img, (bottom_x_start, y_centers[-1]), (bottom_x_end, y_centers[-1]), (0, 0, 255), thickness)
    cv2.line(debug_img, (top_x_start, y_centers[0]), (bottom_x_end, y_centers[-1]), (0, 255, 0), 2)
    cv2.line(debug_img, (top_x_end, y_centers[0]), (bottom_x_start, y_centers[-1]), (0, 255, 0), 2)
    
    if debug:
        display_img = resize_for_display(debug_img)
        cv2.imshow(f"Debug: Linha da Coluna - Ângulo de Cobb = {angle:.2f}°", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return angle, debug_img

# Função principal pra inferir uma imagem
def inferir_imagem(image_path):
    angle, marked_image = detect_cobb_points(image_path)
    if angle is None:
        return None, None, "Erro ao processar a imagem"
    
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    label_map = {0: "Sem escoliose", 1: "Escoliose leve", 2: "Escoliose moderada", 3: "Escoliose grave"}
    classification = label_map[predicted.item()]
    
    print(f"Imagem: {os.path.basename(image_path)}")
    print(f"Ângulo de Cobb: {angle:.2f}°")
    print(f"Classificação: {classification}")
    print()
    
    return angle, marked_image, classification

if __name__ == "__main__":
    # Caminho base para o diretório de dados
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    
    # Caminhos para as pastas de escoliose e sem_escoliose
    escoliose_path = os.path.join(data_path, 'escoliose')
    sem_escoliose_path = os.path.join(data_path, 'sem_escoliose')
    
    # Listas para armazenar todos os caminhos de imagens
    test_images = []
    
    # Adicionar imagens de escoliose
    test_images.extend([
        os.path.join(escoliose_path, img) 
        for img in os.listdir(escoliose_path) 
        if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ])
    
    # Adicionar imagens sem escoliose
    test_images.extend([
        os.path.join(sem_escoliose_path, img) 
        for img in os.listdir(sem_escoliose_path) 
        if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ])
    
    # Processar todas as imagens
    for image in test_images:
        angle, marked_image, classification = inferir_imagem(image)