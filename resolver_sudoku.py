import json
import zipfile
import argparse
import numpy as np
import tensorflow as tf

from dataset import gerar_tabuleiro_completo, aplicar_mascara, codificar_one_hot

def remover_quantization_config(objeto):
    """
    Remove recursivamente o campo 'quantization_config' do arquivo de configuração
    do modelo Keras.

    Isso corrige erro de incompatibilidade entre versões do Keras/TensorFlow.
    """

    if isinstance(objeto, dict):
        objeto.pop("quantization_config", None)

        for valor in objeto.values():
            remover_quantization_config(valor)

    elif isinstance(objeto, list):
        for item in objeto:
            remover_quantization_config(item)


def carregar_modelo_corrigido(caminho_modelo):
    """
    Tenta carregar o modelo normalmente.

    Se der erro por causa do campo 'quantization_config',
    cria uma cópia corrigida do arquivo .keras e carrega essa cópia.
    """

    try:
        return tf.keras.models.load_model(caminho_modelo, compile=False)

    except Exception as erro:
        mensagem_erro = str(erro)

        if "quantization_config" not in mensagem_erro:
            raise erro

        print("\nFoi detectado erro de compatibilidade com 'quantization_config'.")
        print("Criando uma versão corrigida do modelo...")

        caminho_corrigido = caminho_modelo.replace(".keras", "_corrigido.keras")

        with zipfile.ZipFile(caminho_modelo, "r") as arquivo_original:
            with zipfile.ZipFile(caminho_corrigido, "w") as arquivo_corrigido:

                for item in arquivo_original.infolist():
                    conteudo = arquivo_original.read(item.filename)

                    if item.filename == "config.json":
                        config = json.loads(conteudo.decode("utf-8"))

                        remover_quantization_config(config)

                        conteudo = json.dumps(config).encode("utf-8")

                    arquivo_corrigido.writestr(item, conteudo)

        print(f"Modelo corrigido salvo como: {caminho_corrigido}")

        return tf.keras.models.load_model(caminho_corrigido, compile=False)

def imprimir_grade(tabuleiro, titulo="Tabuleiro"):
    """
    Mostra o tabuleiro 4x4 em formato de grade legível.
    """

    print(f"\n{titulo}")
    print("+---+---+---+---+")

    for i in range(4):
        linha = "|"

        for j in range(4):
            valor = tabuleiro[i][j]

            if valor == 0:
                texto = " "
            else:
                texto = str(valor)

            linha += f" {texto} |"

        print(linha)
        print("+---+---+---+---+")


def resolver_sudoku(modelo, tabuleiro_inicial):
    """
    Recebe um modelo treinado e um tabuleiro inicial 4x4.

    O tabuleiro inicial pode ter valores:
    0 = célula vazia
    1, 2, 3, 4 = valores já preenchidos

    Retorna:
    tabuleiro_resolvido: matriz 4x4 com a previsão do modelo.
    """

    tabuleiro_inicial = np.array(tabuleiro_inicial, dtype=int)

    if tabuleiro_inicial.shape != (4, 4):
        raise ValueError("O tabuleiro inicial precisa ter formato 4x4.")

    if np.any(tabuleiro_inicial < 0) or np.any(tabuleiro_inicial > 4):
        raise ValueError("O tabuleiro só pode conter valores de 0 a 4.")

    entrada = codificar_one_hot(tabuleiro_inicial, num_classes=5)

    entrada = entrada.reshape(1, 16, 5)

    previsao = modelo.predict(entrada, verbose=0)

    previsao = previsao[0]

    numeros_previstos = np.argmax(previsao, axis=-1) + 1

    tabuleiro_resolvido = numeros_previstos.reshape(4, 4)

    return tabuleiro_resolvido


def verificar_validade(tabuleiro_resolvido):
    """
    Verifica se o tabuleiro resolvido segue as 3 regras do Sudoku 4x4.

    Regra 1:
    Cada linha deve conter os números 1, 2, 3 e 4 sem repetição.

    Regra 2:
    Cada coluna deve conter os números 1, 2, 3 e 4 sem repetição.

    Regra 3:
    Cada subgrade 2x2 deve conter os números 1, 2, 3 e 4 sem repetição.

    Retorna:
    True, se o tabuleiro for válido.
    False, se o tabuleiro for inválido.
    """

    tabuleiro_resolvido = np.array(tabuleiro_resolvido, dtype=int)

    if tabuleiro_resolvido.shape != (4, 4):
        return False

    esperado = {1, 2, 3, 4}

    # Regra 1: verificar linhas
    for linha in tabuleiro_resolvido:
        if set(linha) != esperado:
            return False

    # Regra 2: verificar colunas
    for coluna in tabuleiro_resolvido.T:
        if set(coluna) != esperado:
            return False

    # Regra 3: verificar subgrades 2x2
    for linha_inicio in [0, 2]:
        for coluna_inicio in [0, 2]:
            subgrade = tabuleiro_resolvido[
                linha_inicio:linha_inicio + 2,
                coluna_inicio:coluna_inicio + 2
            ]

            if set(subgrade.flatten()) != esperado:
                return False

    return True


def verificar_regras_detalhado(tabuleiro_resolvido):
    """
    Mostra no terminal se cada regra foi atendida.
    """

    tabuleiro_resolvido = np.array(tabuleiro_resolvido, dtype=int)

    esperado = {1, 2, 3, 4}

    linhas_validas = True
    colunas_validas = True
    subgrades_validas = True

    for linha in tabuleiro_resolvido:
        if set(linha) != esperado:
            linhas_validas = False

    for coluna in tabuleiro_resolvido.T:
        if set(coluna) != esperado:
            colunas_validas = False

    for linha_inicio in [0, 2]:
        for coluna_inicio in [0, 2]:
            subgrade = tabuleiro_resolvido[
                linha_inicio:linha_inicio + 2,
                coluna_inicio:coluna_inicio + 2
            ]

            if set(subgrade.flatten()) != esperado:
                subgrades_validas = False

    print("\nVerificação das regras:")
    print(f"Regra 1 - Linhas válidas: {linhas_validas}")
    print(f"Regra 2 - Colunas válidas: {colunas_validas}")
    print(f"Regra 3 - Subgrades 2x2 válidas: {subgrades_validas}")

    return linhas_validas, colunas_validas, subgrades_validas


def converter_texto_para_tabuleiro(texto):
    """
    Converte um texto em formato de tabuleiro.

    Exemplo de entrada:
    "0,2,0,4;3,0,1,0;2,0,4,3;0,3,0,1"

    Retorna:
    matriz numpy 4x4.
    """

    linhas = texto.split(";")

    if len(linhas) != 4:
        raise ValueError("O texto precisa ter 4 linhas separadas por ponto e vírgula.")

    tabuleiro = []

    for linha in linhas:
        valores = linha.split(",")

        if len(valores) != 4:
            raise ValueError("Cada linha precisa ter 4 valores separados por vírgula.")

        linha_convertida = [int(valor.strip()) for valor in valores]

        tabuleiro.append(linha_convertida)

    return np.array(tabuleiro, dtype=int)


def gerar_tabuleiro_aleatorio():
    """
    Gera um novo tabuleiro aleatório usando as funções da Pessoa 1.

    Primeiro gera um Sudoku completo válido.
    Depois remove alguns valores para criar o tabuleiro inicial.
    """

    tabuleiro_completo = gerar_tabuleiro_completo()

    tabuleiro_inicial = aplicar_mascara(tabuleiro_completo, num_vazios=8)

    return tabuleiro_inicial, tabuleiro_completo


def main():
    parser = argparse.ArgumentParser(
        description="Resolver Sudoku 4x4 usando o modelo treinado pela Pessoa 2."
    )

    parser.add_argument(
        "--modelo",
        type=str,
        default="modelo_sudoku.keras",
        help="Caminho do modelo treinado."
    )

    parser.add_argument(
        "--tabuleiro",
        type=str,
        default=None,
        help='Tabuleiro manual. Exemplo: "0,2,0,4;3,0,1,0;2,0,4,3;0,3,0,1"'
    )

    args = parser.parse_args()

    print("Carregando modelo treinado...")
    modelo = carregar_modelo_corrigido(args.modelo)

    if args.tabuleiro is None:
        print("Gerando novo tabuleiro aleatório da Pessoa 1...")
        tabuleiro_inicial, tabuleiro_esperado = gerar_tabuleiro_aleatorio()
    else:
        print("Usando tabuleiro informado manualmente...")
        tabuleiro_inicial = converter_texto_para_tabuleiro(args.tabuleiro)
        tabuleiro_esperado = None

    imprimir_grade(tabuleiro_inicial, "Tabuleiro inicial")

    tabuleiro_resolvido = resolver_sudoku(modelo, tabuleiro_inicial)

    imprimir_grade(tabuleiro_resolvido, "Tabuleiro resolvido pelo modelo")

    valido = verificar_validade(tabuleiro_resolvido)

    verificar_regras_detalhado(tabuleiro_resolvido)

    print(f"\nResultado final válido? {valido}")

    if tabuleiro_esperado is not None:
        imprimir_grade(tabuleiro_esperado, "Solução esperada gerada pela Pessoa 1")

        acertos = np.sum(tabuleiro_resolvido == tabuleiro_esperado)
        total = 16
        percentual = acertos / total * 100

        print(f"\nComparação com a solução esperada:")
        print(f"Células corretas: {acertos}/{total}")
        print(f"Acurácia neste tabuleiro: {percentual:.2f}%")


if __name__ == "__main__":
    main()