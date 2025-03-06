import os
import cv2
import numpy as np

# Caminho absoluto das pastas de entrada e saída
base_dir = os.path.dirname(os.path.abspath(__file__))  # Diretório do script atual
input_dirs = [
    os.path.join(base_dir, '..', '..', 'data', 'raw', 'escoliose'),  # Caminho para 'escoliose'
    os.path.join(base_dir, '..', '..', 'data', 'raw', 'sem_escoliose')  # Caminho para 'sem_escoliose'
]

output_dirs = [
    os.path.join(base_dir, '..', '..', 'data', 'processed', 'escoliose'),  # Caminho de saída para 'escoliose'
    os.path.join(base_dir, '..', '..', 'data', 'processed', 'sem_escoliose')  # Caminho de saída para 'sem_escoliose'
]

# Função para garantir que os diretórios de saída existam
def create_output_dirs():
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f'Pasta criada: {output_dir}')
        else:
            print(f'Pasta já existe: {output_dir}')

# Função para pré-processamento das imagens
def preprocess_image(image_path):
    # Lê a imagem
    img = cv2.imread(image_path)

    # Verifica se a imagem foi lida corretamente
    if img is None:
        print(f"Erro ao ler a imagem {image_path}")
        return None

    # Redimensiona a imagem (tamanho fixo para entrada no modelo)
    img_resized = cv2.resize(img, (224, 224))

    # Converte para escala de cinza (opcional dependendo do modelo)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Normaliza a imagem
    img_normalized = img_gray / 255.0

    # Retorna a imagem processada
    return img_normalized

# Função principal de pré-processamento
def process_images():
    create_output_dirs()

    # Processa as imagens de cada diretório
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        if os.path.exists(input_dir):
            print(f'Processando imagens em: {input_dir}')
            for image_name in os.listdir(input_dir):
                image_path = os.path.join(input_dir, image_name)

                if os.path.isfile(image_path):
                    try:
                        # Processa a imagem
                        processed_img = preprocess_image(image_path)

                        # Verifica se a imagem foi processada corretamente
                        if processed_img is not None:
                            # Salva a imagem processada
                            output_path = os.path.join(output_dir, image_name)
                            np.save(output_path, processed_img)  # Usando .npy para salvar o array processado
                            print(f'Imagem processada e salva: {output_path}')
                        else:
                            print(f'Imagem não processada corretamente: {image_name}')

                    except Exception as e:
                        print(f'Erro ao processar a imagem {image_name}: {e}')
        else:
            print(f"Pasta {input_dir} não encontrada.")

if __name__ == "__main__":
    process_images()
