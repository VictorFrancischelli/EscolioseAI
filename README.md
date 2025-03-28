# EscolioseAI

## 📖 Descrição
O **EscolioseAI** é um projeto de análise automática de imagens de raio-X para detecção e classificação de escoliose, utilizando redes neurais convolucionais (CNN) em Python com PyTorch. Além da análise, o sistema permite a exportação de relatórios em PDF com o resultado da análise.

## 📁 Estrutura do Projeto

EscolioseAI/
│── data/                        # Dados de entrada (imagens)
│   ├── escoliose/               # Imagens com escoliose (em formato .jpg.npy)
│   ├── sem_escoliose/           # Imagens sem escoliose (em formato .jpg.npy)
│
│
│── notebooks/                   # Notebooks de experimentação
│
│── src/                         # Código-fonte principal
│   ├── model/                   # Definição e treinamento de modelos
│   │   |── treinamento.py       # Script de treinamento do modelo
|   |   |── inference.py
|   |   |── cnn_classificacao.py
│   |
│   └── ui/                       # Interface gráfica (PySide6)
│
│── .gitignore                   # Arquivos a serem ignorados pelo Git
│── config.yaml                  # Arquivo de configurações gerais
│── main.py                      # Arquivo principal para rodar o sistema
│── modelo_treinado.pth          # Modelo treinado (arquivos de rede neural)
│── README.md                    # Descrição geral do projeto
│── requirements.txt             # Dependências do projeto
│
│── run_app.bat                  # Executa o main.py
│── run_git_update.bat           # Adiciona, faz commit e push de tudo no Git
│── run_train_model.bat          # Executa o script de treinamento (agora dentro de src/model)

---

## 🛠️ Como executar o projeto

### 1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
```

### 2. Entre na pasta do projeto:
```bash
cd EscolioseAI
```

### 3. Crie o ambiente virtual (se ainda não tiver):
```bash
python -m venv venv
```

### 4. Ative o ambiente virtual:
- **No Windows:**
```bash
venv\Scripts\activate
```
- **No Linux/Mac:**
```bash
source venv/bin/activate
```

### 5. Instale as dependências:
```bash
pip install -r requirements.txt
```

### 6. Treine o modelo (opcional se já tiver um modelo salvo):
```bash
python treinamento.py
```

### 7. Rode a aplicação:
```bash
python main.py
```

---

## ✅ Funcionalidades
- Interface gráfica amigável com PySide6
- Análise automática de imagens
- Classificação entre escoliose e sem escoliose
- Exportação de relatório em PDF com imagem analisada
- Armazenamento automático dos relatórios na pasta `exportados`

---

## 👤 Autor
- **Victor Fernando Rodrigues Francischelli**
- GitHub: [seu-usuario](https://github.com/seu-usuario)

---

## 📄 Licença
Este projeto está licenciado sob a licença MIT.


