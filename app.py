from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
from src.model.inference import inferir_imagem

app = FastAPI()

# Criar pastas se não existirem
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Salvar o arquivo de upload
    image_path = os.path.join("uploads", file.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Processar a imagem com o inference.py
    angle, marked_image_path, classification = inferir_imagem(image_path, output_dir="static")
    if angle is None:
        return JSONResponse(status_code=400, content={"error": "Erro ao processar a imagem"})
    
    # Retornar os resultados
    return {
        "angle": angle,
        "image_url": marked_image_path,
        "classification": classification
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)