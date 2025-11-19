# Instruções para Execução dos Experimentos - Trabalho 3

Este documento descreve como compilar e executar os experimentos necessários para o Trabalho 3 (Parte A e Parte B), conforme solicitado na especificação.

## 1. Pré-requisitos

Certifique-se de estar na máquina alvo (`nv00`) ou em um ambiente com GPU NVIDIA configurada e compilador `nvcc` disponível.

Os seguintes arquivos devem estar presentes no diretório raiz:
*   `segmented-sort-bitonic.cu` (Código fonte da Parte A)
*   `mppSort.cu` (Código fonte da Parte B)
*   `compila_partA.sh` (Script de compilação da Parte A)
*   `compila_partB.sh` (Script de compilação da Parte B)
*   `run_experiments.sh` (Script de automação dos experimentos)
*   `my_randomizer.h` (Dependência do gerador de números)

## 2. Executando Tudo Automaticamente

Foi criado um script `run_experiments.sh` que realiza todo o processo: compilação, execução dos testes da Parte A e execução dos benchmarks da Parte B.

Para rodar:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

Os resultados serão exibidos na tela e salvos automaticamente no arquivo `resultados_experimentos.txt`.

## 3. Execução Manual (Passo a Passo)

Caso prefira executar etapa por etapa:

### Passo 3.1: Compilação

**Parte A (Segmented Bitonic Sort):**
Gera binários para 256, 512 e 1024 threads por bloco.
```bash
chmod +x compila_partA.sh
./compila_partA.sh
```

**Parte B (mppSort):**
Gera o binário `mppSort`.
```bash
chmod +x compila_partB.sh
./compila_partB.sh
```

### Passo 3.2: Experimentos Parte A

Execute o binário (recomendado `segmented-sort-bitonic-1024` na `nv00`) com diferentes faixas de segmentos para comparar com o Thrust.

Exemplos:
```bash
# 8M elementos, segmentos entre 20 e 4000
./segmented-sort-bitonic-1024 -n 8000000 -segRange 20 4000

# 8M elementos, segmentos entre 3000 e 4000
./segmented-sort-bitonic-1024 -n 8000000 -segRange 3000 4000

# 8M elementos, segmentos entre 20 e 8000
./segmented-sort-bitonic-1024 -n 8000000 -segRange 20 8000
```

### Passo 3.3: Experimentos Parte B

Execute o `mppSort` variando o tamanho da entrada (1M, 2M, 4M, 8M). O parâmetro `h` (número de bins) deve ser ajustado para garantir que os segmentos caibam na memória compartilhada (aprox. 48KB).

Recomendação de parâmetros (`./mppSort <N> <h> <nR>`):

```bash
# 1 Milhão de elementos
./mppSort 1000000 512 10

# 2 Milhões de elementos
./mppSort 2000000 1024 10

# 4 Milhões de elementos
./mppSort 4000000 2048 10

# 8 Milhões de elementos
./mppSort 8000000 4096 10
```

## 4. Relatório

Utilize os dados gerados em `resultados_experimentos.txt` para preencher a tabela de resultados solicitada no relatório PDF.
