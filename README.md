# Solução de Sudoku 4x4 com Redes Neurais Artificiais (RNA)

Este repositório contém uma Proposta de Solução de uma Rede Neural Artificial (RNA) multicamadas em Python para resolver o quebra-cabeça Sudoku, usando uma grade 4x4 com subgrupos 2x2 preenchidos com números de S = {1, 2, 3, 4}. Desenvolvido como requisito para a disciplina de Inteligência Artificial, ministrada pelo professor Edjard Mota.

## Equipe de Desenvolvimento e Papéis

* **Pessoa 1 (Arquitetura de Dados):** Andrew Donovan Coelho Santos - 22152016
* **Pessoa 2 (Engenharia do Modelo):** Isaque da Silva Targino - 22352193
* **Pessoa 3 (Inferência e Validação):** Sabrina Amorim da Penha - 22152026
* **Pessoa 4 (Integração e DevOps):** Anderson Oliveira de Araújo - 22152023

---

## Arquitetura do Projeto e Pipeline

O projeto foi modularizado para garantir a escalabilidade e a clareza do fluxo de dados:

1. `dataset.py`: Responsável pela geração de tabuleiros 4x4 válidos, aplicação de máscaras (buracos) e codificação *One-Hot*.
2. `modelo.py`: Define a arquitetura da RNA (camadas densas) e gerencia o treinamento.
3. `resolver_sudoku.py`: Módulo de inferência que carrega o modelo, valida as restrições lógicas e formata as saídas.
4. `main.py`: Pipeline integrador que unifica todas as etapas em uma única execução.

### Como Executar

O projeto utiliza Python 3.11. Siga os passos:

```bash
# 1. Crie e ative um ambiente virtual
python3 -m venv venv
source venv/bin/activate  # No Windows: .\venv\Scripts\activate

# 2. Instale as dependências
pip install numpy tensorflow

# 3. Execute o pipeline unificado
python main.py
```
## Discussão Teórica (Análise Crítica)

### 1. Generalização para a Dimensão N × N
Resolver o Sudoku 4x4 com uma rede Multilayer Perceptron (MLP) é possível devido ao espaço de estados reduzido. No entanto, escalar para grades N×N (como o Sudoku 9x9) apresenta desafios arquiteturais complexos.

### 1.1 Mudanças Necessárias para Generalização
Parametrização Dinâmica: O modelo precisaria aceitar entradas variáveis N² e adaptar a camada de saída para comportar N classes por célula.

Mudança Arquitetural: Redes densas (MLPs) ignoram a estrutura espacial do tabuleiro. Para N×N, seriam necessárias Redes Neurais Convolucionais (CNNs) para capturar dependências locais (subgrades) ou Redes Neurais em Grafos (GNNs), que são mais adequadas para representar as restrições de adjacência de um Sudoku.

### 1.2 Dificuldades na Implementação
A principal barreira é a natureza estocástica das RNAs. O Sudoku é um problema lógico de restrições rígidas (hard constraints). Enquanto uma RNA aprende padrões estatísticos, ela não aprende, por padrão, que números não podem se repetir. Escalar para N×N causaria uma falha generalizada na satisfação das regras, exigindo abordagens híbridas (Neuro-Simbólicas) que incorporem solucionadores lógicos (como SAT solvers) dentro do processo de predição.

### 2. O Problema da Abordagem "Generate and Test"
A abordagem "Generate and Test" consiste em gerar uma solução candidata (populando o tabuleiro) para, posteriormente, validar se ela cumpre as regras. Esta técnica é ineficiente para problemas de grande escala por ser "cega":

Custo Computacional: A maior parte do tempo é gasta validando estados matematicamente impossíveis.

Falta de Propagação: A inteligência artificial, para resolver Sudokus complexos, precisa de Propagação de Restrições. Em vez de gerar e testar, o modelo deve descartar opções inválidas durante o preenchimento, reduzindo o espaço de busca. O "Generate and Test" puro não explora essas deduções, tratando o Sudoku como uma mera adivinhação, o que é o oposto da eficiência lógica exigida em problemas CSP (Constraint Satisfaction Problems).
