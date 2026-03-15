# =============================================================================
# CONFIGURAÇÕES DE ESTILO E FUNÇÕES AUXILIARES
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

def configurar_estilo(usar_arial=False):
    """
    Configura o estilo dos gráficos conforme manual USP/ESALQ
    """
    sns.set_style("white")
    plt.rcParams['axes.grid'] = False
    if usar_arial:
        plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10

# Paletas de cores
CORES_VIRIDIS = sns.color_palette("viridis", 3)
CORES_CLUSTERS = ['#FF0000', '#0000FF', '#FFFF00']  # vermelho, azul, amarelo