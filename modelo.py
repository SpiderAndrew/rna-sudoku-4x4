import numpy as np
import tensorflow as tf
from tensorflow import keras
from dataset import gerar_conjunto_dados, codificar_one_hot

# =============================================================================
# FLUXO DOS DADOS:
#   Entrada (X): tabuleiro com buracos → shape (16, 5)
#     Cada uma das 16 células é um vetor one-hot de 5 posições:
#     [1,0,0,0,0] = vazio, [0,1,0,0,0] = 1, [0,0,1,0,0] = 2, etc.
#
#   Saída (y): tabuleiro completo → shape (16, 4)
#     A rede prevê UMA distribuição de probabilidade por célula,
#     com 4 posições (uma pra cada número 1-4).
#     Ex: [0.02, 0.95, 0.02, 0.01] → a rede "acha" que é o número 2.
#
#   Importante: a saída tem 4 classes, não 5.
#   O "vazio" (classe 0) não é uma resposta válida — descartamos ele do y.
# =============================================================================

def construir_modelo_rna(input_shape=(16, 5)):
    """
    Monta a arquitetura da rede neural sequencial (camada por camada).

    Parâmetro:
        input_shape: formato da entrada — (16 células, 5 classes each)

    Retorna:
        modelo Keras compilável, ainda sem treino.
    """
    modelo = keras.Sequential([

        # INPUT: recebe o tabuleiro codificado em one-hot
        keras.layers.Input(shape=input_shape),

        # FLATTEN: transforma a matriz (16, 5) num vetor plano de 80 valores.
        # A rede densa precisa de um vetor 1D, não de uma matriz.
        keras.layers.Flatten(),

        # CAMADAS OCULTAS: aprendem padrões internos do Sudoku.
        # Dense(N) = N neurônios, todos conectados à camada anterior.
        # ReLU (Rectified Linear Unit): função de ativação que "zera" valores
        # negativos. É o padrão pra camadas ocultas — simples e eficaz.
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),  # camada maior no meio = mais capacidade
        keras.layers.Dense(128, activation='relu'),

        # SAÍDA: 16 células × 4 números possíveis = 64 neurônios de saída.
        # Sem ativação aqui — o Reshape e Softmax cuidam disso a seguir.
        keras.layers.Dense(16 * 4),

        # RESHAPE: reorganiza os 64 valores em (16 células, 4 classes).
        # Facilita aplicar o Softmax por célula individualmente.
        keras.layers.Reshape((16, 4)),

        # SOFTMAX: converte os valores brutos em probabilidades (somam 1.0).
        # axis=-1 = aplica por célula (não pro tabuleiro inteiro).
        # Ex: [1.2, 3.1, 0.5, 0.8] → [0.07, 0.82, 0.04, 0.07]
        #      → a rede "aposta" no número 2 (índice 1 → número 2)
        keras.layers.Softmax(axis=-1)
    ])

    return modelo


def treinar_modelo(modelo, X, y, epochs=50, batch_size=32):
    """
    Treina o modelo com os dados gerados

    Parâmetros:
        modelo: arquitetura construída por construir_modelo_rna()
        X: tabuleiros com buracos, shape (N, 16, 5)
        y: tabuleiros completos, shape (N, 16, 5)
        epochs: quantas vezes a rede passa por todos os dados
        batch_size: quantos exemplos ela processa de uma vez

    Retorna:
        modelo treinado + histórico de métricas
    """

    # y vem com 5 classes (0=vazio, 1, 2, 3, 4).
    # Removemos a coluna 0 porque a rede não precisa "prever vazio" —
    # ela só precisa saber qual número (1-4) vai em cada célula.
    # y[:, :, 1:] = todas as amostras, todas as células, colunas 1 até 4.
    y_treino = y[:, :, 1:]  # (N, 16, 5) → (N, 16, 4)

    modelo.compile(
        # ADAM: otimizador que ajusta os pesos da rede a cada batch.
        # É o padrão moderno — adapta a taxa de aprendizado automaticamente.
        optimizer='adam',

        # CATEGORICAL CROSSENTROPY: mede o erro quando a saída é
        # uma distribuição de probabilidade (que é exatamente o nosso caso).
        # Quanto mais confiante a rede na resposta errada, maior a penalidade.
        loss='categorical_crossentropy',

        # ACCURACY: % de células onde a classe com maior probabilidade
        # coincide com a classe correta.
        metrics=['accuracy']
    )

    historico = modelo.fit(
        X, y_treino,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # 20% dos dados ficam de fora pro teste de validação
        verbose=1
    )

    return modelo, historico


def decodificar_previsao(previsao):
    """
    Converte a saída da rede (probabilidades) de volta pra números 1-4.

    A rede retorna shape (16, 4) — probabilidades por célula.
    argmax pega o índice de maior probabilidade.
    Somamos +1 porque índice 0 = número 1, índice 1 = número 2, etc.
    """
    return np.argmax(previsao, axis=-1) + 1  # shape (16,) com valores 1-4


def verificar_sudoku_valido(tabuleiro):
    """
    Verifica se um tabuleiro 4x4 é um Sudoku válido.
    Checa as 3 regras: linhas, colunas e subgrupos 2x2.

    Retorna True se válido, False se inválido.
    """
    esperado = {1, 2, 3, 4}

    # Regra 1: cada linha tem os 4 números sem repetição
    for linha in tabuleiro:
        if set(linha) != esperado:
            return False

    # Regra 2: cada coluna tem os 4 números sem repetição
    for col in tabuleiro.T:
        if set(col) != esperado:
            return False

    # Regra 3: cada subgrupo 2x2 tem os 4 números sem repetição
    for i in [0, 2]:
        for j in [0, 2]:
            subgrupo = tabuleiro[i:i+2, j:j+2].flatten()
            if set(subgrupo) != esperado:
                return False

    return True


def avaliar_modelo_real(modelo, num_testes=200):
    """
    Avaliação real do modelo: gera tabuleiros novos (nunca vistos),
    pede pra rede resolver, e verifica se o resultado é um Sudoku válido.

    Isso vai além da 'accuracy' do Keras, que mede célula por célula.
    Aqui medimos tabuleiros completos e corretos pelas regras do jogo.
    """
    from dataset import gerar_tabuleiro_completo, aplicar_mascara

    tabuleiros_validos = 0
    celulas_corretas_total = 0
    total_celulas = 0

    for _ in range(num_testes):
        # Gera um par problema/solução novo
        alvo = gerar_tabuleiro_completo()
        problema = aplicar_mascara(alvo, num_vazios=8)

        # Codifica e passa pela rede
        entrada = codificar_one_hot(problema).reshape(1, 16, 5)
        previsao = modelo.predict(entrada, verbose=0)
        resultado = decodificar_previsao(previsao[0]).reshape(4, 4)

        # Checa acurácia célula a célula
        corretas = np.sum(resultado == alvo)
        celulas_corretas_total += corretas
        total_celulas += 16

        # Checa se o tabuleiro inteiro é Sudoku válido
        if verificar_sudoku_valido(resultado):
            tabuleiros_validos += 1

    acc_celulas = celulas_corretas_total / total_celulas
    acc_tabuleiros = tabuleiros_validos / num_testes

    print(f"\n{'='*50}")
    print(f"AVALIAÇÃO REAL — {num_testes} tabuleiros novos")
    print(f"{'='*50}")
    print(f"Acurácia por célula:      {acc_celulas:.2%}")
    print(f"Tabuleiros 100% corretos: {acc_tabuleiros:.2%}  ({tabuleiros_validos}/{num_testes})")
    print(f"{'='*50}")

    return acc_tabuleiros


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("Gerando dados...")
    X, y = gerar_conjunto_dados(num_amostras=1000)

    print("\nConstruindo modelo...")
    modelo = construir_modelo_rna(input_shape=(16, 5))
    modelo.summary()

    print("\nTreinando...")
    modelo, historico = treinar_modelo(modelo, X, y, epochs=50)

    print("\nTreinamento concluído!")
    acc_final = historico.history['accuracy'][-1]
    val_acc_final = historico.history['val_accuracy'][-1]
    print(f"Acurácia treino:    {acc_final:.2%}")
    print(f"Acurácia validação: {val_acc_final:.2%}")

    # Avaliação real com verificação das regras do Sudoku
    avaliar_modelo_real(modelo, num_testes=200)

    modelo.save("modelo_sudoku.keras")
    print("\nModelo salvo em modelo_sudoku.keras")