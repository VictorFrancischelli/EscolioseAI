from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QPushButton
from src.ui.tela_analise import TelaAnalise
from src.ui.tela_historico import TelaHistorico


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EscolioseAI")
        self.setGeometry(100, 100, 600, 400)
        
        self.central_widget = QStackedWidget(self)
        self.setCentralWidget(self.central_widget)

        # Inicializa as telas
        self.tela_analise = TelaAnalise()
        self.tela_historico = TelaHistorico()

        # Adiciona as telas no QStackedWidget
        self.central_widget.addWidget(self.tela_analise)
        self.central_widget.addWidget(self.tela_historico)

        # Criação do botão para navegar entre telas
        self.botao_analise = QPushButton("Ir para Análise", self)
        self.botao_analise.clicked.connect(self.mudar_para_analise)
        self.botao_analise.setGeometry(10, 10, 150, 40)
        self.botao_analise.setParent(self)

        self.botao_historico = QPushButton("Ir para Histórico", self)
        self.botao_historico.clicked.connect(self.mudar_para_historico)
        self.botao_historico.setGeometry(170, 10, 150, 40)
        self.botao_historico.setParent(self)

    def mudar_para_analise(self):
        self.central_widget.setCurrentWidget(self.tela_analise)

    def mudar_para_historico(self):
        self.central_widget.setCurrentWidget(self.tela_historico)

# Execução do app
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
