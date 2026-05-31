import numpy as np
import random

def gerar_tabuleiro_completo():
    """
    Gera um tabuleiro Sudoku 4x4 válido (y_train).
    Usa um tabuleiro base e embaralha os símbolos para gerar variações válidas.
    """
    # Matriz base que respeita todas as regras do Sudoku 4x4
    base = np.array([
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1]
    ])

    # Para gerar variações, embaralhamos os números de 1 a 4
    numeros = [1, 2, 3, 4]
    random.shuffle(numeros)

    # Cria um dicionário de mapeamento (ex: 1 vira 3, 2 vira 4, etc.)
    mapeamento = {i + 1: numeros[i] for i in range(4)}

    # Aplica o mapeamento na matriz base
    novo_tabuleiro = np.vectorize(mapeamento.get)(base)
    return novo_tabuleiro

def aplicar_mascara(tabuleiro, num_vazios=8):
    """
    Remove números aleatórios do tabuleiro para criar o problema (X_train).
    O número 0 representará os espaços vazios.
    """
    tabuleiro_mascarado = np.copy(tabuleiro)

    # Transforma a matriz 4x4 em um array plano de 16 posições para facilitar o sorteio
    flat = tabuleiro_mascarado.flatten()

    # Escolhe índices aleatórios para zerar (sem repetição)
    indices_vazios = np.random.choice(16, num_vazios, replace=False)
    flat[indices_vazios] = 0

    # Volta para o formato 4x4
    return flat.reshape(4, 4)

def codificar_one_hot(tabuleiro, num_classes=5):
    """
    Converte o tabuleiro em formato One-Hot Encoding.
    Classes: 0 (vazio), 1, 2, 3 e 4.
    Uma matriz 4x4 (16 valores) vai virar (16, 5), onde cada número vira um vetor de zeros e um.
    """
    flat = tabuleiro.flatten()
    # O np.eye cria uma matriz identidade. Usamos o array 'flat' como índice
    # para gerar os vetores one-hot instantaneamente.
    one_hot = np.eye(num_classes)[flat]
    return one_hot

def gerar_conjunto_dados(num_amostras=1000):
    """
    Une as funções anteriores para gerar o conjunto completo de treino.
    X_train: Tabuleiros com zeros (codificados em one-hot).
    y_train: Tabuleiros completos (codificados em one-hot).
    """
    X_train = []
    y_train = []

    for _ in range(num_amostras):
        alvo = gerar_tabuleiro_completo()
        problema = aplicar_mascara(alvo, num_vazios=random.randint(6, 10)) # De 6 a 10 buracos

        # A rede neural precisa dos dados de entrada (X) e saída (y) em one-hot
        X_train.append(codificar_one_hot(problema, num_classes=5))
        y_train.append(codificar_one_hot(alvo, num_classes=5))

    return np.array(X_train), np.array(y_train)

# ==========================================
# TESTE DAS FUNÇÕES
# ==========================================
if __name__ == "__main__":
    print("--- 1. Gerando Tabuleiro Completo (Alvo) ---")
    alvo = gerar_tabuleiro_completo()
    print(alvo)

    print("\n--- 2. Aplicando Máscara (Problema) ---")
    problema = aplicar_mascara(alvo, num_vazios=6)
    print(problema)

    print("\n--- 3. Exemplo de One-Hot Encoding (Problema) ---")
    problema_oh = codificar_one_hot(problema)
    print(f"Formato da matriz (Células, Classes): {problema_oh.shape}")
    print("Primeira célula em one-hot (ex: se for 0, o vetor é [1. 0. 0. 0. 0.]):")
    print(problema_oh[0])

    print("\n--- 4. Gerando Lote de Treinamento ---")
    X, y = gerar_conjunto_dados(num_amostras=100)
    print(f"X_train shape: {X.shape}") # Esperado: (100, 16, 5)
    print(f"y_train shape: {y.shape}") # Esperado: (100, 16, 5)