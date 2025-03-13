import sys
from PySide6.QtWidgets import QApplication
from tela_analise import TelaAnalise

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TelaAnalise()
    window.setWindowTitle("Análise de Escoliose")
    window.show()
    sys.exit(app.exec())
