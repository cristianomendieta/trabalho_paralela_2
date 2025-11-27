#!/bin/bash

# Script para executar os benchmarks de referência (binários do professor)
# Autor: Cristiano Mendieta (Gerado por GitHub Copilot)

OUTPUT_FILE="resultados_referencia.txt"
MACHINE=$(hostname)

echo "=== Iniciando Benchmarks de Referência (Professor) ===" | tee $OUTPUT_FILE
echo "Data: $(date)" | tee -a $OUTPUT_FILE
echo "Máquina: $MACHINE" | tee -a $OUTPUT_FILE
echo "======================================================" | tee -a $OUTPUT_FILE

# Diretório onde estão os binários do professor
REF_DIR="copia3-publica"

# Verificar se o diretório existe
if [ ! -d "$REF_DIR" ]; then
    echo "ERRO: Diretório $REF_DIR não encontrado." | tee -a $OUTPUT_FILE
    exit 1
fi

# Definir qual binário usar (assumindo 1024 como padrão para comparação justa)
BIN_REF="$REF_DIR/segmented-sort-bitonic-1024"

if [ -f "$BIN_REF" ]; then
    # Dar permissão de execução se necessário
    chmod +x "$BIN_REF"

    echo "" | tee -a $OUTPUT_FILE
    echo ">>> Executando Binário de Referência: $BIN_REF" | tee -a $OUTPUT_FILE
    
    # Caso 1: Muitos segmentos pequenos/médios
    echo "--- Teste Ref.1: 8M elementos, segmentos entre 20 e 4000 ---" | tee -a $OUTPUT_FILE
    "$BIN_REF" -n 8000000 -segRange 20 4000 | tee -a $OUTPUT_FILE
    echo "" | tee -a $OUTPUT_FILE

    # Caso 2: Segmentos maiores
    echo "--- Teste Ref.2: 8M elementos, segmentos entre 3000 e 4000 ---" | tee -a $OUTPUT_FILE
    "$BIN_REF" -n 8000000 -segRange 3000 4000 | tee -a $OUTPUT_FILE
    echo "" | tee -a $OUTPUT_FILE

    # Caso 3: Segmentos muito variados
    echo "--- Teste Ref.3: 8M elementos, segmentos entre 20 e 8000 ---" | tee -a $OUTPUT_FILE
    "$BIN_REF" -n 8000000 -segRange 20 8000 | tee -a $OUTPUT_FILE
    echo "" | tee -a $OUTPUT_FILE

else
    echo "ERRO: Binário $BIN_REF não encontrado." | tee -a $OUTPUT_FILE
    echo "Tentando listar o diretório $REF_DIR:" | tee -a $OUTPUT_FILE
    ls -l "$REF_DIR" | tee -a $OUTPUT_FILE
fi

echo "=== Benchmarks de Referência Concluídos ===" | tee -a $OUTPUT_FILE
echo "Resultados salvos em: $OUTPUT_FILE"
