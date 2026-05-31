import os
import argparse


from dataset import gerar_conjunto_dados
from modelo import construir_modelo_rna, treinar_modelo
from resolver_sudoku import carregar_modelo_corrigido, gerar_tabuleiro_aleatorio, imprimir_grade, resolver_sudoku, verificar_validade, verificar_regras_detalhado

def executar_pipeline_completo():
    """
    Executa o pipeline completo:
    1. Geração de Dados (Pessoa 1)
    2. Construção e Treinamento da RNA (Pessoa 2)
    3. Inferência e Validação (Pessoa 3)
    """
    print("="*60)
    print("INICIANDO PIPELINE: SUDOKU 4x4 COM REDES NEURAIS")
    print("="*60)


    caminho_modelo = "modelo_sudoku.keras"

    if not os.path.exists(caminho_modelo):
        print("\n[INFO] Modelo não encontrado. Iniciando treinamento do zero...")
        print("[INFO] Gerando dataset de treinamento...")
        X, y = gerar_conjunto_dados(num_amostras=2000)

        print("[INFO] Construindo arquitetura da RNA...")
        modelo = construir_modelo_rna(input_shape=(16, 5))

        print("[INFO] Treinando o modelo...")
        modelo, historico = treinar_modelo(modelo, X, y, epochs=50)
        modelo.save(caminho_modelo)
        print(f"[INFO] Modelo salvo com sucesso em {caminho_modelo}!")
    else:
        print(f"\n[INFO] Modelo {caminho_modelo} já existe. Pulando etapa de treinamento.")
        modelo = carregar_modelo_corrigido(caminho_modelo)


    print("\n" + "="*60)
    print("FASE DE INFERÊNCIA: RESOLVENDO UM NOVO SUDOKU")
    print("="*60)


    tabuleiro_inicial, tabuleiro_esperado = gerar_tabuleiro_aleatorio()

    imprimir_grade(tabuleiro_inicial, "Tabuleiro Inicial (Problema Gerado)")


    tabuleiro_resolvido = resolver_sudoku(modelo, tabuleiro_inicial)

    imprimir_grade(tabuleiro_resolvido, "Solução Encontrada pela RNA")
    imprimir_grade(tabuleiro_esperado, "Gabarito Real (Ground Truth)")


    print("\n[INFO] Verificando satisfação das restrições lógicas...")
    valido = verificar_validade(tabuleiro_resolvido)
    verificar_regras_detalhado(tabuleiro_resolvido)

    if valido:
        print("\n[RESULTADO] SUCESSO! A rede neural gerou um Sudoku válido!")
    else:
        print("\n[RESULTADO] FALHA! O tabuleiro gerado viola regras do Sudoku.")

if __name__ == "__main__":
    executar_pipeline_completo()