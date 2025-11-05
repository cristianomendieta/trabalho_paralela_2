#!/bin/bash

# Script de teste automatizado para mppSort
# Executa os experimentos solicitados no relatório

echo "=========================================="
echo "  Testes Automáticos - mppSort GPU"
echo "=========================================="
echo ""

# Verificar se o executável existe
if [ ! -f "./mppSort" ]; then
    echo "ERRO: Executável mppSort não encontrado!"
    echo "Execute ./compila.sh primeiro"
    exit 1
fi

# Verificar se CUDA está disponível
if ! command -v nvidia-smi &> /dev/null; then
    echo "AVISO: nvidia-smi não encontrado. Você está em uma máquina com GPU?"
    read -p "Deseja continuar mesmo assim? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Mostrar informações da GPU
echo "Informações da GPU:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || echo "Não foi possível obter informações da GPU"
echo ""

# Parâmetros
H=256        # Número de bins
NR=10        # Número de repetições

# Criar diretório para resultados
RESULTS_DIR="resultados_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Resultados serão salvos em: $RESULTS_DIR"
echo ""

# Função para executar teste
run_test() {
    local n_elements=$1
    local label=$2
    
    echo "=========================================="
    echo "Teste: $label elementos"
    echo "=========================================="
    echo "Parâmetros: nElements=$n_elements, h=$H, nR=$NR"
    echo ""
    
    OUTPUT_FILE="$RESULTS_DIR/test_${label}.txt"
    
    # Executar o programa e salvar saída
    ./mppSort $n_elements $H $NR | tee "$OUTPUT_FILE"
    
    # Verificar se executou com sucesso
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "✓ Teste concluído com sucesso!"
        echo "  Resultados salvos em: $OUTPUT_FILE"
    else
        echo ""
        echo "✗ ERRO ao executar teste!"
        echo "  Verifique o arquivo: $OUTPUT_FILE"
    fi
    
    echo ""
    echo "Pressione ENTER para continuar..."
    read
}

# Executar os testes conforme especificação do relatório
# Nota: M = 10^6 (não potências de 2)

echo "Iniciando bateria de testes..."
echo ""
sleep 2

# Teste 1: 1M elementos
run_test 1000000 "1M"

# Teste 2: 2M elementos
run_test 2000000 "2M"

# Teste 3: 4M elementos
run_test 4000000 "4M"

# Teste 4: 8M elementos
run_test 8000000 "8M"

# Resumo final
echo "=========================================="
echo "  TODOS OS TESTES CONCLUÍDOS!"
echo "=========================================="
echo ""
echo "Resultados salvos em: $RESULTS_DIR/"
echo ""
echo "Arquivos gerados:"
ls -lh "$RESULTS_DIR/"
echo ""

# Extrair e mostrar resumo de performance
echo "=========================================="
echo "  RESUMO DE PERFORMANCE"
echo "=========================================="
echo ""

for file in "$RESULTS_DIR"/test_*.txt; do
    if [ -f "$file" ]; then
        test_name=$(basename "$file" .txt | sed 's/test_//')
        echo "--- $test_name ---"
        
        # Extrair vazão do mppSort
        mppsort_throughput=$(grep "Throughput:" "$file" | head -1 | awk '{print $2}')
        
        # Extrair vazão do Thrust
        thrust_throughput=$(grep "Throughput:" "$file" | tail -1 | awk '{print $2}')
        
        # Extrair speedup
        speedup=$(grep "mppSort vs Thrust:" "$file" | awk '{print $4}')
        
        # Extrair verificação
        verification=$(grep -E "(Ordenação correta|ERRO NA ORDENAÇÃO)" "$file")
        
        echo "  mppSort: ${mppsort_throughput:-N/A} GElements/s"
        echo "  Thrust:  ${thrust_throughput:-N/A} GElements/s"
        echo "  Speedup: ${speedup:-N/A}"
        echo "  Verificação: ${verification:-N/A}"
        echo ""
    fi
done

# Criar arquivo CSV com resumo para facilitar criação da tabela do relatório
CSV_FILE="$RESULTS_DIR/resumo.csv"
echo "Nro_Elementos,Vazao_mppSort_GElements_s,Vazao_Thrust_GElements_s,Speedup" > "$CSV_FILE"

for file in "$RESULTS_DIR"/test_*.txt; do
    if [ -f "$file" ]; then
        test_name=$(basename "$file" .txt | sed 's/test_//')
        
        # Determinar número de elementos
        case $test_name in
            "1M") n_elem="1000000" ;;
            "2M") n_elem="2000000" ;;
            "4M") n_elem="4000000" ;;
            "8M") n_elem="8000000" ;;
            *) n_elem="N/A" ;;
        esac
        
        # Extrair dados
        mppsort_throughput=$(grep "Throughput:" "$file" | head -1 | awk '{print $2}')
        thrust_throughput=$(grep "Throughput:" "$file" | tail -1 | awk '{print $2}')
        speedup=$(grep "mppSort vs Thrust:" "$file" | awk '{print $4}' | sed 's/x//')
        
        # Adicionar ao CSV
        echo "$n_elem,$mppsort_throughput,$thrust_throughput,$speedup" >> "$CSV_FILE"
    fi
done

echo "=========================================="
echo "Resumo CSV criado em: $CSV_FILE"
echo "Use este arquivo para preencher a tabela do relatório!"
echo "=========================================="
echo ""

# Mostrar conteúdo do CSV
echo "Conteúdo do CSV:"
cat "$CSV_FILE"
echo ""

echo "Testes finalizados. Boa sorte com o relatório!"
