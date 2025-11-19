#!/bin/bash

# Script para executar os experimentos do Trabalho 3 (Parte A e Parte B)
# Autor: Cristiano Mendieta (Gerado por GitHub Copilot)

# Configurações
OUTPUT_FILE="resultados_experimentos.txt"
MACHINE=$(hostname)

echo "=== Iniciando Experimentos do Trabalho 3 ===" | tee $OUTPUT_FILE
echo "Data: $(date)" | tee -a $OUTPUT_FILE
echo "Máquina: $MACHINE" | tee -a $OUTPUT_FILE
echo "============================================" | tee -a $OUTPUT_FILE

# -----------------------------------------------------------------------------
# 1. Compilação
# -----------------------------------------------------------------------------
echo "" | tee -a $OUTPUT_FILE
echo ">>> Fase 1: Compilação" | tee -a $OUTPUT_FILE

# Compilar Parte A (Segmented Bitonic Sort)
echo "Compilando Parte A (Segmented Bitonic Sort)..." | tee -a $OUTPUT_FILE
if [ -f "compila_partA.sh" ]; then
    chmod +x compila_partA.sh
    ./compila_partA.sh >> compilation_log.txt 2>&1
    if [ $? -eq 0 ]; then
        echo "Parte A compilada com sucesso." | tee -a $OUTPUT_FILE
    else
        echo "ERRO: Falha na compilação da Parte A. Verifique compilation_log.txt." | tee -a $OUTPUT_FILE
    fi
else
    echo "ERRO: Script compila_partA.sh não encontrado." | tee -a $OUTPUT_FILE
fi

# Compilar Parte B (mppSort)
echo "Compilando Parte B (mppSort)..." | tee -a $OUTPUT_FILE
if [ -f "compila_partB.sh" ]; then
    chmod +x compila_partB.sh
    ./compila_partB.sh >> compilation_log.txt 2>&1
    if [ $? -eq 0 ]; then
        echo "Parte B compilada com sucesso." | tee -a $OUTPUT_FILE
    else
        echo "ERRO: Falha na compilação da Parte B. Verifique compilation_log.txt." | tee -a $OUTPUT_FILE
    fi
else
    echo "ERRO: Script compila_partB.sh não encontrado." | tee -a $OUTPUT_FILE
fi

# -----------------------------------------------------------------------------
# 2. Experimentos Parte A
# -----------------------------------------------------------------------------
echo "" | tee -a $OUTPUT_FILE
echo ">>> Fase 2: Experimentos Parte A (Segmented Bitonic Sort)" | tee -a $OUTPUT_FILE
echo "Comparando desempenho com Thrust para diferentes configurações de segmentos." | tee -a $OUTPUT_FILE

# Definir qual binário usar (assumindo 1024 como padrão ou o melhor)
# Se estiver na nv00, idealmente testaria todos, mas vamos usar o 1024 para o relatório principal
BIN_PART_A="./segmented-sort-bitonic-1024"

if [ -f "$BIN_PART_A" ]; then
    echo "Usando binário: $BIN_PART_A" | tee -a $OUTPUT_FILE
    
    # Caso 1: Muitos segmentos pequenos/médios
    echo "--- Teste A.1: 8M elementos, segmentos entre 20 e 4000 ---" | tee -a $OUTPUT_FILE
    $BIN_PART_A -n 8000000 -segRange 20 4000 | tee -a $OUTPUT_FILE
    echo "" | tee -a $OUTPUT_FILE

    # Caso 2: Segmentos maiores
    echo "--- Teste A.2: 8M elementos, segmentos entre 3000 e 4000 ---" | tee -a $OUTPUT_FILE
    $BIN_PART_A -n 8000000 -segRange 3000 4000 | tee -a $OUTPUT_FILE
    echo "" | tee -a $OUTPUT_FILE

    # Caso 3: Segmentos muito variados
    echo "--- Teste A.3: 8M elementos, segmentos entre 20 e 8000 ---" | tee -a $OUTPUT_FILE
    $BIN_PART_A -n 8000000 -segRange 20 8000 | tee -a $OUTPUT_FILE
    echo "" | tee -a $OUTPUT_FILE
else
    echo "ERRO: Binário $BIN_PART_A não encontrado." | tee -a $OUTPUT_FILE
fi

# -----------------------------------------------------------------------------
# 3. Experimentos Parte B
# -----------------------------------------------------------------------------
echo "" | tee -a $OUTPUT_FILE
echo ">>> Fase 3: Experimentos Parte B (mppSort Integrado)" | tee -a $OUTPUT_FILE
echo "Executando mppSort para 1M, 2M, 4M e 8M elementos." | tee -a $OUTPUT_FILE

BIN_PART_B="./mppSort"
NR=10 # Número de repetições para média

if [ -f "$BIN_PART_B" ]; then
    # Definir h (número de bins) para garantir que os segmentos caibam na Shared Memory (48KB)
    # Max elementos por segmento ~ 12k uints.
    # Para 8M, h=2048 => média 4k elementos (seguro).
    
    # 1M Elementos
    echo "--- Teste B.1: 1M Elementos (h=512) ---" | tee -a $OUTPUT_FILE
    $BIN_PART_B 1000000 512 $NR | tee -a $OUTPUT_FILE
    echo "" | tee -a $OUTPUT_FILE

    # 2M Elementos
    echo "--- Teste B.2: 2M Elementos (h=1024) ---" | tee -a $OUTPUT_FILE
    $BIN_PART_B 2000000 1024 $NR | tee -a $OUTPUT_FILE
    echo "" | tee -a $OUTPUT_FILE

    # 4M Elementos
    echo "--- Teste B.3: 4M Elementos (h=2048) ---" | tee -a $OUTPUT_FILE
    $BIN_PART_B 4000000 2048 $NR | tee -a $OUTPUT_FILE
    echo "" | tee -a $OUTPUT_FILE

    # 8M Elementos
    echo "--- Teste B.4: 8M Elementos (h=4096) ---" | tee -a $OUTPUT_FILE
    $BIN_PART_B 8000000 4096 $NR | tee -a $OUTPUT_FILE
    echo "" | tee -a $OUTPUT_FILE

else
    echo "ERRO: Binário $BIN_PART_B não encontrado." | tee -a $OUTPUT_FILE
fi

echo "=== Experimentos Concluídos ===" | tee -a $OUTPUT_FILE
echo "Resultados salvos em: $OUTPUT_FILE"
