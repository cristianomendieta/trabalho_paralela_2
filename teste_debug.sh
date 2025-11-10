#!/bin/bash

# Teste minúsculo para debug
echo "=============================================="
echo "  TESTE DEBUG - 1000 elementos apenas"
echo "=============================================="
echo ""

# Compilar
if [ ! -f "./mppSort" ] || [ "mppSort.cu" -nt "./mppSort" ]; then
    echo "Recompilando..."
    ./compila.sh
    if [ $? -ne 0 ]; then
        echo "ERRO na compilação!"
        exit 1
    fi
    echo ""
fi

# Teste super pequeno
echo "Teste: 1000 elementos, 16 bins, 1 repetição"
echo "./mppSort 1000 16 1"
echo ""
./mppSort 1000 16 1

echo ""
if [ $? -eq 0 ]; then
    echo "✓ Teste passou!"
else
    echo "✗ Teste falhou!"
fi
