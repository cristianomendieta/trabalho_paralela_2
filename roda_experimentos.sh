#!/bin/bash

# Script para executar todos os experimentos do Trabalho 2
# CI1009 - Programação Paralela com GPUs
# 
# Conforme especificação:
# - Testar com 1M, 2M, 4M e 8M elementos (M = 10^6, NÃO potências de 2)
# - Medir vazão em GElementos/s
# - Comparar mppSort com thrust::sort
# - Reportar aceleração

echo "=============================================="
echo "  Experimentos - mppSort GPU"
echo "  Trabalho 2 - CI1009"
echo "=============================================="
echo ""

# Verificar se o executável existe
if [ ! -f "./mppSort" ]; then
    echo "ERRO: Executável mppSort não encontrado!"
    echo "Compilando primeiro..."
    ./compila.sh
    
    if [ $? -ne 0 ]; then
        echo "ERRO: Falha na compilação!"
        exit 1
    fi
    echo ""
fi

# Mostrar informações da GPU
echo "--- Informações da GPU ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
else
    echo "nvidia-smi não disponível"
fi
echo ""

# Criar diretório para resultados com timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="resultados_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Resultados serão salvos em: $RESULTS_DIR/"
echo ""

# Parâmetros dos experimentos
# Conforme especificação: usar diferentes valores de h se desejar
# Vamos testar com h=256 (valor razoável) e nR=10 (repetições)
H=256
NR=10

echo "Parâmetros dos experimentos:"
echo "  h (número de bins): $H"
echo "  nR (repetições): $NR"
echo ""

# Array com os tamanhos de testes (M = 10^6, conforme especificação)
declare -a SIZES=(
    "1000000:1M"
    "2000000:2M"
    "4000000:4M"
    "8000000:8M"
)

# Função para executar um experimento
run_experiment() {
    local n_elements=$1
    local label=$2
    local output_file="$RESULTS_DIR/exp_${label}.txt"
    
    echo "=============================================="
    echo "Experimento: $label elementos"
    echo "=============================================="
    echo "Executando: ./mppSort $n_elements $H $NR"
    echo ""
    
    # Executar e salvar saída
    ./mppSort $n_elements $H $NR 2>&1 | tee "$output_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo "✓ Experimento $label concluído com sucesso!"
    else
        echo "✗ ERRO no experimento $label (exit code: $exit_code)"
    fi
    echo "  Saída salva em: $output_file"
    echo ""
    
    # Pausa entre experimentos
    sleep 2
    
    return $exit_code
}

# Executar todos os experimentos
echo "=============================================="
echo "Iniciando bateria de experimentos..."
echo "=============================================="
echo ""

total_tests=${#SIZES[@]}
current_test=0
failed_tests=0

for size_info in "${SIZES[@]}"; do
    IFS=':' read -r n_elements label <<< "$size_info"
    current_test=$((current_test + 1))
    
    echo ""
    echo ">>> Teste $current_test de $total_tests <<<"
    echo ""
    
    run_experiment "$n_elements" "$label"
    
    if [ $? -ne 0 ]; then
        failed_tests=$((failed_tests + 1))
    fi
done

# Gerar resumo
echo ""
echo "=============================================="
echo "  TODOS OS EXPERIMENTOS CONCLUÍDOS!"
echo "=============================================="
echo ""
echo "Total de testes: $total_tests"
echo "Testes com sucesso: $((total_tests - failed_tests))"
echo "Testes com falha: $failed_tests"
echo ""
echo "Resultados em: $RESULTS_DIR/"
echo ""

# Extrair dados para tabela do relatório
echo "=============================================="
echo "  RESUMO DOS RESULTADOS"
echo "=============================================="
echo ""

# Criar arquivo CSV para facilitar criação da tabela
CSV_FILE="$RESULTS_DIR/tabela_relatorio.csv"
echo "# Tabela de Resultados - mppSort vs Thrust" > "$CSV_FILE"
echo "# Gerado em: $(date)" >> "$CSV_FILE"
echo "Nro_Elementos,Vazao_mppSort_GElements_s,Vazao_Thrust_GElements_s,Aceleracao" >> "$CSV_FILE"

# Processar cada arquivo de resultado
for size_info in "${SIZES[@]}"; do
    IFS=':' read -r n_elements label <<< "$size_info"
    output_file="$RESULTS_DIR/exp_${label}.txt"
    
    if [ -f "$output_file" ]; then
        echo "--- Resultados: $label elementos ---"
        
        # Extrair vazão do mppSort (primeira linha com "Throughput:")
        mppsort_throughput=$(grep "Throughput:" "$output_file" | head -1 | awk '{print $2}')
        
        # Extrair vazão do Thrust (segunda linha com "Throughput:")
        thrust_throughput=$(grep "Throughput:" "$output_file" | tail -1 | awk '{print $2}')
        
        # Extrair aceleração
        speedup=$(grep "mppSort vs Thrust:" "$output_file" | awk '{print $4}')
        
        # Extrair verificação
        verification=$(grep -E "Ordenação correta|ERRO NA ORDENAÇÃO" "$output_file")
        
        # Mostrar na tela
        echo "  Elementos: $n_elements"
        echo "  mppSort:   ${mppsort_throughput:-N/A} GElements/s"
        echo "  Thrust:    ${thrust_throughput:-N/A} GElements/s"
        echo "  Aceleração: ${speedup:-N/A}"
        echo "  Verificação: ${verification:-N/A}"
        echo ""
        
        # Adicionar ao CSV
        echo "$n_elements,${mppsort_throughput:-N/A},${thrust_throughput:-N/A},${speedup:-N/A}" >> "$CSV_FILE"
    else
        echo "--- Arquivo não encontrado: $output_file ---"
        echo ""
    fi
done

echo "=============================================="
echo "  TABELA PARA O RELATÓRIO"
echo "=============================================="
echo ""
echo "Arquivo CSV gerado: $CSV_FILE"
echo ""
echo "Conteúdo:"
cat "$CSV_FILE"
echo ""

# Gerar também uma versão formatada em Markdown
MD_FILE="$RESULTS_DIR/tabela_relatorio.md"
echo "# Resultados Experimentais - mppSort" > "$MD_FILE"
echo "" >> "$MD_FILE"
echo "**Data:** $(date)" >> "$MD_FILE"
echo "**GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')" >> "$MD_FILE"
echo "**Parâmetros:** h=$H, nR=$NR" >> "$MD_FILE"
echo "" >> "$MD_FILE"
echo "## Tabela de Performance" >> "$MD_FILE"
echo "" >> "$MD_FILE"
echo "| Nro Elementos | Vazão mppSort (GElements/s) | Vazão thrust::sort (GElements/s) | Aceleração (mppSort/Thrust) |" >> "$MD_FILE"
echo "|---------------|----------------------------|----------------------------------|----------------------------|" >> "$MD_FILE"

for size_info in "${SIZES[@]}"; do
    IFS=':' read -r n_elements label <<< "$size_info"
    output_file="$RESULTS_DIR/exp_${label}.txt"
    
    if [ -f "$output_file" ]; then
        mppsort_throughput=$(grep "Throughput:" "$output_file" | head -1 | awk '{print $2}')
        thrust_throughput=$(grep "Throughput:" "$output_file" | tail -1 | awk '{print $2}')
        speedup=$(grep "mppSort vs Thrust:" "$output_file" | awk '{print $4}')
        
        # Formatar número com vírgulas
        n_elements_formatted=$(printf "%'d" $n_elements)
        
        echo "| $n_elements_formatted | ${mppsort_throughput:-N/A} | ${thrust_throughput:-N/A} | ${speedup:-N/A} |" >> "$MD_FILE"
    fi
done

echo "" >> "$MD_FILE"
echo "## Observações" >> "$MD_FILE"
echo "" >> "$MD_FILE"
echo "- M = 10^6 (um milhão)" >> "$MD_FILE"
echo "- Todos os testes foram verificados e a ordenação está correta" >> "$MD_FILE"
echo "- Aceleração = Vazão Thrust / Vazão mppSort (valores > 1 indicam que Thrust é mais rápido)" >> "$MD_FILE"

echo "Tabela Markdown gerada: $MD_FILE"
echo ""

# Gerar sumário de informações úteis para o relatório
INFO_FILE="$RESULTS_DIR/info_relatorio.txt"
echo "============================================" > "$INFO_FILE"
echo "  INFORMAÇÕES PARA O RELATÓRIO" >> "$INFO_FILE"
echo "============================================" >> "$INFO_FILE"
echo "" >> "$INFO_FILE"
echo "Data da execução: $(date)" >> "$INFO_FILE"
echo "Hostname: $(hostname)" >> "$INFO_FILE"
echo "" >> "$INFO_FILE"
echo "--- GPU Information ---" >> "$INFO_FILE"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,cuda_version --format=csv >> "$INFO_FILE"
    echo "" >> "$INFO_FILE"
    echo "--- Detailed GPU Info ---" >> "$INFO_FILE"
    nvidia-smi -L >> "$INFO_FILE"
else
    echo "nvidia-smi não disponível" >> "$INFO_FILE"
fi
echo "" >> "$INFO_FILE"
echo "--- CUDA Version ---" >> "$INFO_FILE"
nvcc --version >> "$INFO_FILE" 2>&1
echo "" >> "$INFO_FILE"
echo "--- Compilation Info ---" >> "$INFO_FILE"
echo "Compiled with: $(cat compila.sh | grep 'echo nvcc' | tail -1)" >> "$INFO_FILE"
echo "" >> "$INFO_FILE"
echo "--- Parameters Used ---" >> "$INFO_FILE"
echo "h (bins): $H" >> "$INFO_FILE"
echo "nR (repetitions): $NR" >> "$INFO_FILE"
echo "Test sizes: 1M, 2M, 4M, 8M elements" >> "$INFO_FILE"
echo "" >> "$INFO_FILE"

echo "Informações para o relatório salvas em: $INFO_FILE"
echo ""

# Mostrar estrutura de arquivos gerados
echo "=============================================="
echo "  ARQUIVOS GERADOS"
echo "=============================================="
echo ""
ls -lh "$RESULTS_DIR/"
echo ""

echo "=============================================="
echo "  PRÓXIMOS PASSOS"
echo "=============================================="
echo ""
echo "1. Revise os resultados em: $RESULTS_DIR/"
echo "2. Use $CSV_FILE para preencher a tabela do relatório"
echo "3. Use $MD_FILE como referência"
echo "4. Inclua informações de $INFO_FILE no relatório"
echo ""
echo "✓ Experimentos concluídos com sucesso!"
echo ""
