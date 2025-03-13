from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QPushButton
from PySide6.QtCore import Qt
import os

class TelaHistorico(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histórico de Análises")
        self.setGeometry(100, 100, 400, 300)
        
        layout = QVBoxLayout()
        
        self.label = QLabel("Histórico de Análises")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        
        self.lista_historico = QListWidget()
        layout.addWidget(self.lista_historico)
        
        self.botao_fechar = QPushButton("Fechar")
        self.botao_fechar.clicked.connect(self.close)
        layout.addWidget(self.botao_fechar)
        
        self.setLayout(layout)
        
        self.carregar_historico()
    
    def carregar_historico(self):
        historico_path = "data/historico/historico.txt"
        if os.path.exists(historico_path):
            with open(historico_path, "r") as file:
                linhas = file.readlines()
                self.lista_historico.addItems([linha.strip() for linha in linhas])
        else:
            self.lista_historico.addItem("Nenhum histórico encontrado.")
