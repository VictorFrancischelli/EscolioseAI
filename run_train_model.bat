@echo off
echo ================================
echo  Ativando ambiente virtual...
echo ================================
call venv\Scripts\activate

echo ================================
echo  Iniciando treinamento do modelo
echo ================================
python -m src.model.train

echo ================================
echo  Treinamento concluído!
echo  O modelo treinado foi salvo na raiz do projeto.
echo ================================

pause
