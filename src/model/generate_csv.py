import os
import csv
import numpy as np
import cv2

# Função pra calcular o ângulo de Cobb
def detect_cobb_points(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, img
    
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
        return None, None, img
    
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
        return None, None, img
    
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
    return None, None, angle, original_img

# Função pra classificar a gravidade
def classify_scoliosis(angle):
    if angle <= 10:
        return "sem_escoliose"
    elif 10 < angle <= 25:
        return "leve"
    elif 25 < angle <= 44:
        return "moderada"
    else:
        return "grave"

# Função pra gerar o dataset
def generate_dataset(folder_path, output_csv):
    data = []
    
    for folder in ['escoliose', 'sem_escoliose']:
        full_folder_path = os.path.join(folder_path, folder)
        for file_name in os.listdir(full_folder_path):
            if file_name.endswith('.jpg'):
                image_path = os.path.join(full_folder_path, file_name)
                _, _, angle, _ = detect_cobb_points(image_path)
                if angle is not None:
                    classification = classify_scoliosis(angle)
                    data.append([image_path, classification])
                    print(f"{file_name}: Ângulo = {angle:.2f}°, Classificação = {classification}")
                else:
                    print(f"Não foi possível processar {file_name}")
    
    # Salvar no CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'label'])
        writer.writerows(data)
    print(f"Dataset salvo em {output_csv}")

# Executar
if __name__ == "__main__":
    # Caminhos relativos à pasta src/model/
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    output_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset.csv'))
    generate_dataset(folder_path, output_csv)