@echo off
set /p msg=Digite a mensagem do commit: 
echo ================================
echo Adicionando arquivos ao Git...
echo ================================
git add .

echo ================================
echo Fazendo commit...
echo ================================
git commit -m "%msg%"

echo ================================
echo Enviando para o GitHub...
echo ================================
git push

echo ================================
echo Commit e push concluídos!
echo ================================

pause