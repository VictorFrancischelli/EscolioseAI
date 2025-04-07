from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import sys
import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

# Adicionar o diretório src/model ao sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'model'))
from cnn_classificacao import CNNModel

app = FastAPI(title="EscolioseAI API Desktop")

# Carregar o modelo treinado
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'modelo_treinado.pth'))
model = CNNModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Funções auxiliares
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def detect_cobb_points(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagem inválida")

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
        raise HTTPException(status_code=400, detail="Falha na detecção da coluna")

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
        raise HTTPException(status_code=400, detail="Falha nas inclinações")

    top_slope, top_start_idx = max(top_slopes, key=lambda x: abs(x[0]))
    bottom_slope, bottom_start_idx = max(bottom_slopes, key=lambda x: abs(x[0]))  # Corrigido para bottom_slopes

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
    start_point = (int(spine_line_x[0]), int(y_centers[0]))
    end_point = (int(spine_line_x[-1]), int(y_centers[-1]))
    cv2.line(debug_img, start_point, end_point, (0, 0, 255), 2)
    line_length = 200
    thickness = 4
    top_x_start = int(spine_line_x[0] - line_length // 2)
    top_x_end = int(spine_line_x[0] + line_length // 2)
    cv2.line(debug_img, (top_x_start, int(y_centers[0])), (top_x_end, int(y_centers[0])), (0, 0, 255), thickness)
    bottom_x_start = int(spine_line_x[-1] - line_length // 2)
    bottom_x_end = int(spine_line_x[-1] + line_length // 2)
    cv2.line(debug_img, (bottom_x_start, int(y_centers[-1])), (bottom_x_end, int(y_centers[-1])), (0, 0, 255), thickness)
    cv2.line(debug_img, (top_x_start, int(y_centers[0])), (bottom_x_end, int(y_centers[-1])), (0, 255, 0), 2)
    cv2.line(debug_img, (top_x_end, int(y_centers[0])), (bottom_x_start, int(y_centers[-1])), (0, 255, 0), 2)

    marked_path = os.path.join("temp", "marked_" + os.path.basename(image_path))
    os.makedirs("temp", exist_ok=True)
    cv2.imwrite(marked_path, debug_img)

    return angle, marked_path

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        temp_path = os.path.join("temp", file.filename)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        angle, marked_path = detect_cobb_points(temp_path)
        input_tensor = preprocess_image(temp_path)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

        classification = {0: "Sem escoliose", 1: "Escoliose leve", 2: "Escoliose moderada", 3: "Escoliose grave"}[predicted.item()]

        return JSONResponse(content={
            "angle": float(angle) if angle else 0,
            "classification": classification,
            "marked_image_path": marked_path
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)