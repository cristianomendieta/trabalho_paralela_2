#!/bin/bash

# Script para gerar resultados CSV - mppSort
# Trabalho 2 - CI1009 Programação Paralela com GPUs
# UFPR - Universidade Federal do Paraná
# Autores: Cristiano Mendieta e Thiago Ruiz
# Data: Novembro de 2025

echo "=============================================="
echo "  Gerando Resultados CSV - mppSort"
echo "=============================================="
echo ""

# Verificar executável
if [ ! -f "./mppSort" ]; then
    echo "Compilando mppSort..."
    ./compila.sh || exit 1
fi

# Parâmetros
H=256
NR=10

# Criar diretório
RESULTS_DIR="resultados_finais_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Arquivo principal
OUTPUT_CSV="$RESULTS_DIR/resultados_mppsort.csv"

# Executar um teste para obter informações de configuração
temp_info="$RESULTS_DIR/temp_info.txt"
./mppSort 1000 $H 1 > "$temp_info" 2>&1

# Extrair número de blocos e threads
NB=$(grep "Number of blocks" "$temp_info" | awk '{print $5}' | tr -d '()')
NT=$(grep "Threads per block" "$temp_info" | awk '{print $5}' | tr -d '()')

# Cabeçalho
cat > "$OUTPUT_CSV" << EOF
# Resultados Experimentais - mppSort GPU
# Trabalho 2 - CI1009 Programacao Paralela com GPUs
# Data: $(date)
# GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')
# Parametros: h=$H bins, nR=$NR repeticoes, nb=$NB blocos, nt=$NT threads/bloco
# M = 10^6 (conforme especificacao)

EOF

echo "Nro_Elementos,Vazao_mppSort_GElements_s,Vazao_Thrust_GElements_s,Speedup,Tempo_Medio_ms,Blocos,Threads_Bloco" >> "$OUTPUT_CSV"

# Função para executar e extrair dados
run_test() {
    local n=$1
    local label=$2
    
    echo "Executando teste: $label ($n elementos)..."
    
    local temp="$RESULTS_DIR/temp_${label}.txt"
    ./mppSort $n $H $NR > "$temp" 2>&1
    
    if [ $? -eq 0 ]; then
        # Extrair dados
        local mppsort_thr=$(grep "Throughput:" "$temp" | head -1 | awk '{print $2}')
        local thrust_thr=$(grep "Throughput:" "$temp" | tail -1 | awk '{print $2}')
        local speedup=$(grep "mppSort vs Thrust:" "$temp" | awk '{print $4}')
        local avg_time=$(grep "Average time per iteration:" "$temp" | awk '{print $5}')
        
        # Adicionar ao CSV
        echo "$n,$mppsort_thr,$thrust_thr,$speedup,$avg_time,$NB,$NT" >> "$OUTPUT_CSV"
        
        echo "  ✓ Vazão mppSort: $mppsort_thr GElements/s"
        echo "  ✓ Vazão Thrust: $thrust_thr GElements/s"
        echo "  ✓ Speedup: $speedup"
        echo "  ✓ Configuração: $NB blocos x $NT threads"
        echo ""
        
        return 0
    else
        echo "$n,ERRO,ERRO,ERRO,ERRO,$NB,$NT" >> "$OUTPUT_CSV"
        echo "  ✗ ERRO na execução"
        echo ""
        return 1
    fi
}

# Executar testes conforme especificação
echo "Iniciando experimentos (1M, 2M, 4M, 8M)..."
echo ""

run_test 1000000 "1M"
run_test 2000000 "2M"
run_test 4000000 "4M"
run_test 8000000 "8M"

# Criar tabela formatada para o relatório
RELATORIO_MD="$RESULTS_DIR/tabela_para_relatorio.md"

cat > "$RELATORIO_MD" << EOF
# Tabela de Resultados para o Relatório

## Experimentos mppSort em GPU

**Configuração:**
- Bins (h): $H
- Repetições (nR): $NR
- Blocos (nb): $NB
- Threads por bloco (nt): $NT
- Tamanhos testados: 1M, 2M, 4M, 8M elementos (M = 10^6)

## Tabela de Performance

| Nro Elementos | Vazão mppSort<br/>(GElements/s) | Vazão thrust::sort<br/>(GElements/s) | Speedup<br/>(Thrust/mppSort) |
|:-------------:|:-------------------------------:|:------------------------------------:|:---------------------------:|
EOF

# Adicionar dados à tabela
while IFS=',' read -r n mppsort thrust speedup time nb nt; do
    # Pular linhas de comentário e cabeçalho
    [[ "$n" =~ ^#.*$ ]] && continue
    [[ "$n" == "Nro_Elementos" ]] && continue
    
    # Formatar número com separador de milhares
    n_formatted=$(printf "%'d" $n 2>/dev/null || echo $n)
    
    echo "| $n_formatted | $mppsort | $thrust | $speedup |" >> "$RELATORIO_MD"
done < "$OUTPUT_CSV"

cat >> "$RELATORIO_MD" << 'EOF'

## Interpretação dos Resultados

- **Vazão (Throughput)**: Medida em GElements/s (bilhões de elementos por segundo)
- **Speedup**: Razão entre vazão do Thrust e vazão do mppSort
  - Speedup < 1: mppSort é mais rápido
  - Speedup > 1: Thrust é mais rápido
  - Speedup ≈ 1: Performance similar

## Observações

- M = 10^6 (não são potências de 2, conforme especificação)
- Todos os testes incluem verificação automática de corretude
- Medições realizadas com sincronização CUDA entre kernels
- Tempo inclui todos os 5 kernels: histogram, scans, partition e sorting

EOF

# Resumo final
echo "=============================================="
echo "  RESULTADOS GERADOS COM SUCESSO"
echo "=============================================="
echo ""
echo "Arquivos criados:"
echo "  1. $OUTPUT_CSV"
echo "     - Dados em formato CSV"
echo ""
echo "  2. $RELATORIO_MD"
echo "     - Tabela formatada para o relatório"
echo ""
echo "Conteúdo do CSV:"
echo "----------------------------------------"
cat "$OUTPUT_CSV"
echo "----------------------------------------"
echo ""
