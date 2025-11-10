#!/bin/bash

# Script para gerar resultados em formato CSV similar ao exemplo
# Baseado na especificação do trabalho e exemplo_planilha.csv

echo "=============================================="
echo "  Experimentos mppSort - Formato CSV"
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

# Criar diretório para resultados
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="resultados_csv_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Resultados serão salvos em: $RESULTS_DIR/"
echo ""

# Parâmetros conforme especificação
H=256      # Número de bins
NR=10      # Número de repetições

# Array com os tamanhos de testes (M = 10^6)
declare -a SIZES=(
    "1000000:1M"
    "2000000:2M"
    "4000000:4M"
    "8000000:8M"
)

# Arquivo CSV principal
CSV_FILE="$RESULTS_DIR/resultados_detalhados.csv"

# Executar teste rápido para obter configuração
temp_config="$RESULTS_DIR/temp_config.txt"
./mppSort 1000 $H 1 > "$temp_config" 2>&1

# Extrair informações de configuração
NB=$(grep "Number of blocks" "$temp_config" | awk '{print $5}' | tr -d '()')
NT=$(grep "Threads per block" "$temp_config" | awk '{print $5}' | tr -d '()')

echo "# Resultados Experimentais - mppSort" > "$CSV_FILE"
echo "# Data: $(date)" >> "$CSV_FILE"
echo "# GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')" >> "$CSV_FILE"
echo "# Parametros: h=$H bins, nR=$NR repeticoes, nb=$NB blocos, nt=$NT threads/bloco" >> "$CSV_FILE"
echo "" >> "$CSV_FILE"

# Função para executar um experimento e extrair tempos individuais
run_experiment_csv() {
    local n_elements=$1
    local label=$2
    local temp_file="$RESULTS_DIR/temp_${label}.txt"
    
    echo "=============================================="
    echo "Experimento: $label elementos ($n_elements)"
    echo "=============================================="
    echo ""
    
    # Executar o programa e capturar saída
    ./mppSort $n_elements $H $NR > "$temp_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Teste $label concluído"
        
        # Extrair tempos e throughput
        local avg_time=$(grep "Average time per iteration:" "$temp_file" | awk '{print $5}')
        local throughput_mppsort=$(grep "Throughput:" "$temp_file" | head -1 | awk '{print $2}')
        local throughput_thrust=$(grep "Throughput:" "$temp_file" | tail -1 | awk '{print $2}')
        local speedup=$(grep "mppSort vs Thrust:" "$temp_file" | awk '{print $4}')
        
        echo "  Tempo médio: $avg_time ms"
        echo "  Throughput mppSort: $throughput_mppsort GElements/s"
        echo "  Throughput Thrust: $throughput_thrust GElements/s"
        echo "  Speedup: $speedup"
        
        # Adicionar ao CSV no formato do exemplo
        echo "" >> "$CSV_FILE"
        echo "Executando,$NR,vezes,com,$n_elements,elementos,$H,bins,$NB,blocos,$NT,threads" >> "$CSV_FILE"
        echo "mppSort,Tempo_Medio_ms,Throughput_GElements_s,Thrust_Throughput_GElements_s,Speedup" >> "$CSV_FILE"
        echo "Resultados,$avg_time,$throughput_mppsort,$throughput_thrust,$speedup" >> "$CSV_FILE"
        
    else
        echo "✗ ERRO no teste $label"
        echo "" >> "$CSV_FILE"
        echo "Executando,$NR,vezes,com,$n_elements,elementos,e,$H,bins" >> "$CSV_FILE"
        echo "ERRO,na,execucao" >> "$CSV_FILE"
    fi
    
    echo ""
    return $exit_code
}

# Executar todos os experimentos
echo "Iniciando bateria de experimentos..."
echo ""

total_tests=${#SIZES[@]}
current_test=0
failed_tests=0

for size_info in "${SIZES[@]}"; do
    IFS=':' read -r n_elements label <<< "$size_info"
    current_test=$((current_test + 1))
    
    echo ">>> Teste $current_test de $total_tests: $label <<<"
    echo ""
    
    run_experiment_csv "$n_elements" "$label"
    
    if [ $? -ne 0 ]; then
        failed_tests=$((failed_tests + 1))
    fi
    
    sleep 1
done

# Gerar tabela resumo (formato especificação)
SUMMARY_CSV="$RESULTS_DIR/tabela_relatorio.csv"

echo "# Tabela de Resultados - Formato Especificacao" > "$SUMMARY_CSV"
echo "# Comparacao mppSort vs Thrust" >> "$SUMMARY_CSV"
echo "Nro_Elementos,Vazao_mppSort_GElements_s,Vazao_Thrust_GElements_s,Aceleracao_mppSort_Thrust" >> "$SUMMARY_CSV"

for size_info in "${SIZES[@]}"; do
    IFS=':' read -r n_elements label <<< "$size_info"
    temp_file="$RESULTS_DIR/temp_${label}.txt"
    
    if [ -f "$temp_file" ]; then
        throughput_mppsort=$(grep "Throughput:" "$temp_file" | head -1 | awk '{print $2}')
        throughput_thrust=$(grep "Throughput:" "$temp_file" | tail -1 | awk '{print $2}')
        speedup=$(grep "mppSort vs Thrust:" "$temp_file" | awk '{print $4}')
        
        echo "$n_elements,$throughput_mppsort,$throughput_thrust,$speedup" >> "$SUMMARY_CSV"
    else
        echo "$n_elements,ERRO,ERRO,ERRO" >> "$SUMMARY_CSV"
    fi
done

# Gerar tabela formatada em Markdown
MD_TABLE="$RESULTS_DIR/tabela_relatorio.md"

echo "# Resultados Experimentais - mppSort GPU" > "$MD_TABLE"
echo "" >> "$MD_TABLE"
echo "**Data:** $(date)" >> "$MD_TABLE"
echo "**GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')" >> "$MD_TABLE"
echo "**Parâmetros:** h=$H bins, nR=$NR repetições, nb=$NB blocos, nt=$NT threads/bloco" >> "$MD_TABLE"
echo "" >> "$MD_TABLE"
echo "## Tabela de Performance" >> "$MD_TABLE"
echo "" >> "$MD_TABLE"
echo "| Nro Elementos | Vazão mppSort (GElements/s) | Vazão thrust::sort (GElements/s) | Aceleração |" >> "$MD_TABLE"
echo "|---------------|----------------------------|----------------------------------|-----------|" >> "$MD_TABLE"

for size_info in "${SIZES[@]}"; do
    IFS=':' read -r n_elements label <<< "$size_info"
    temp_file="$RESULTS_DIR/temp_${label}.txt"
    
    if [ -f "$temp_file" ]; then
        throughput_mppsort=$(grep "Throughput:" "$temp_file" | head -1 | awk '{print $2}')
        throughput_thrust=$(grep "Throughput:" "$temp_file" | tail -1 | awk '{print $2}')
        speedup=$(grep "mppSort vs Thrust:" "$temp_file" | awk '{print $4}')
        
        n_formatted=$(printf "%'d" $n_elements)
        echo "| $n_formatted | $throughput_mppsort | $throughput_thrust | $speedup |" >> "$MD_TABLE"
    else
        n_formatted=$(printf "%'d" $n_elements)
        echo "| $n_formatted | ERRO | ERRO | ERRO |" >> "$MD_TABLE"
    fi
done

echo "" >> "$MD_TABLE"
echo "## Observações" >> "$MD_TABLE"
echo "" >> "$MD_TABLE"
echo "- M = 10^6 (um milhão)" >> "$MD_TABLE"
echo "- Aceleração = Vazão Thrust / Vazão mppSort" >> "$MD_TABLE"
echo "  - Valores < 1: mppSort é mais rápido que Thrust" >> "$MD_TABLE"
echo "  - Valores > 1: Thrust é mais rápido que mppSort" >> "$MD_TABLE"
echo "- Verificação de corretude realizada automaticamente" >> "$MD_TABLE"

# Resumo final
echo ""
echo "=============================================="
echo "  EXPERIMENTOS CONCLUÍDOS"
echo "=============================================="
echo ""
echo "Total de testes: $total_tests"
echo "Testes com sucesso: $((total_tests - failed_tests))"
echo "Testes com falha: $failed_tests"
echo ""
echo "Arquivos gerados:"
echo "  - $CSV_FILE (resultados detalhados)"
echo "  - $SUMMARY_CSV (tabela resumo)"
echo "  - $MD_TABLE (tabela formatada)"
echo ""

# Mostrar conteúdo da tabela resumo
echo "=============================================="
echo "  TABELA RESUMO"
echo "=============================================="
cat "$SUMMARY_CSV"
echo ""

echo "=============================================="
echo "  ARQUIVOS NO DIRETÓRIO"
echo "=============================================="
ls -lh "$RESULTS_DIR/"
echo ""

echo "✓ Experimentos finalizados!"
echo "Use os arquivos CSV para o relatório"
