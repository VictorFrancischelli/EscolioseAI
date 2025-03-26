@echo off
echo ================================
echo  Ativando ambiente virtual...
echo ================================
call venv\Scripts\activate

echo ================================
echo  Iniciando aplicação EscolioseAI
echo ================================
python main.py

echo ================================
echo  Aplicação finalizada.
echo ================================

pause