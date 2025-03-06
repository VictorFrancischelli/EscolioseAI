import os
import shutil
from sklearn.model_selection import train_test_split

# Caminhos das pastas
raw_dir = 'C:/Users/Victo/OneDrive/Documentos/EscolioseAI/data/raw'
train_dir = 'C:/Users/Victo/OneDrive/Documentos/EscolioseAI/data/train'
val_dir = 'C:/Users/Victo/OneDrive/Documentos/EscolioseAI/data/val'

# Função para dividir os dados
def split_data(image_dir, train_dir, val_dir, test_size=0.2):
    # Cria as pastas de treino e validação, caso não existam
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Lista os arquivos de imagem
    images = os.listdir(image_dir)
    
    # Divide os arquivos em treinamento e validação
    train_files, val_files = train_test_split(images, test_size=test_size, random_state=42)
    
    # Move as imagens para as pastas correspondentes
    for file in train_files:
        shutil.move(os.path.join(image_dir, file), os.path.join(train_dir, file))
    
    for file in val_files:
        shutil.move(os.path.join(image_dir, file), os.path.join(val_dir, file))

# Divida os dados para 'escoliose' e 'sem_escoliose'
split_data(os.path.join(raw_dir, 'escoliose'), os.path.join(train_dir, 'escoliose'), os.path.join(val_dir, 'escoliose'))
split_data(os.path.join(raw_dir, 'sem_escoliose'), os.path.join(train_dir, 'sem_escoliose'), os.path.join(val_dir, 'sem_escoliose'))

print("Divisão dos dados concluída com sucesso!")
