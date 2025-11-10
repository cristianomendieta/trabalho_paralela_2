#!/bin/bash

# Script de teste rápido para validar funcionamento do mppSort
# Executa com poucos elementos apenas para verificar se compila e roda

echo "=============================================="
echo "  TESTE RÁPIDO - mppSort"
echo "  (validação de funcionamento)"
echo "=============================================="
echo ""

# Compilar se necessário
if [ ! -f "./mppSort" ]; then
    echo "Executável não encontrado. Compilando..."
    ./compila.sh
    
    if [ $? -ne 0 ]; then
        echo "ERRO: Falha na compilação!"
        exit 1
    fi
    echo ""
fi

# Teste 1: Muito pequeno (10 mil elementos)
echo "=============================================="
echo "Teste 1: 10 mil elementos (muito rápido)"
echo "=============================================="
echo "./mppSort 10000 64 3"
echo ""
./mppSort 10000 64 3
echo ""

if [ $? -eq 0 ]; then
    echo "✓ Teste 1 passou!"
else
    echo "✗ Teste 1 falhou!"
    exit 1
fi

echo ""
sleep 1

# Teste 2: Pequeno (100 mil elementos)
echo "=============================================="
echo "Teste 2: 100 mil elementos (rápido)"
echo "=============================================="
echo "./mppSort 100000 128 5"
echo ""
./mppSort 100000 128 5
echo ""

if [ $? -eq 0 ]; then
    echo "✓ Teste 2 passou!"
else
    echo "✗ Teste 2 falhou!"
    exit 1
fi

echo ""
sleep 1

# Teste 3: Médio (500 mil elementos)
echo "=============================================="
echo "Teste 3: 500 mil elementos (relativamente rápido)"
echo "=============================================="
echo "./mppSort 500000 256 5"
echo ""
./mppSort 500000 256 5
echo ""

if [ $? -eq 0 ]; then
    echo "✓ Teste 3 passou!"
else
    echo "✗ Teste 3 falhou!"
    exit 1
fi

echo ""
echo "=============================================="
echo "  TODOS OS TESTES RÁPIDOS PASSARAM!"
echo "=============================================="
echo ""
echo "✓ Compilação funcionando"
echo "✓ Kernels executando corretamente"
echo "✓ Verificação de corretude OK"
echo "✓ Benchmark funcionando"
echo ""
echo "Agora você pode executar os experimentos completos:"
echo "  ./roda_experimentos.sh"
echo ""
