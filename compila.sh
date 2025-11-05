#!/bin/bash

# Compilation script for mppSort CUDA implementation
# Based on thrust-sort/compila.sh pattern
# Usage: ./compila.sh

if [ "$(hostname)" = "orval" ]; then
   echo "Compilacao especial na maquina orval"

   #compilação específica para GTX 750ti (máquina orval)
   #OBS:
   # nesse semestre a orval está com cuda 11.8 
   # nessa versao o nvcc NAO suporta gcc 12 ou g++ 12, 
   #   que é o gcc/g++ atualmente na orval
   # entao, apesar disso, consegui compilar com o gcc 12
   #   forçando o uso do gcc-12 conforme abaixo
   echo nvcc -arch sm_50 --allow-unsupported-compiler -ccbin /usr/bin/g++-12 mppSort.cu -o mppSort
   nvcc -arch sm_50 --allow-unsupported-compiler -ccbin /usr/bin/g++-12 mppSort.cu -o mppSort

elif [ "$(hostname)" = "nv00" ]; then

   echo "Compilacao especial na maquina nv00"
   echo "----- compilando especificamente para a GTX 1080ti  (sm_61)"
   echo "nvcc -arch sm_61 -I /usr/include/c++/10 -I /usr/lib/cuda/include/ mppSort.cu -o mppSort"
   nvcc -arch sm_61 -I /usr/include/c++/10 -I /usr/lib/cuda/include/ mppSort.cu -o mppSort

else

   #OBS para compilar para qualquer GPU basta retirar o -arch
   #    mas isso pode deixar a compilacao (ou a carga do programa) mais lenta

   echo compilando para maquina genérica \(`hostname`\)
   #compilação para diversas GPUs
   echo nvcc -O3 mppSort.cu -o mppSort
   nvcc -O3 mppSort.cu -o mppSort

fi

if [ $? -eq 0 ]; then
    echo ""
    echo "Compilacao bem sucedida!"
    echo "Executar com: ./mppSort <nTotalElements> <h> <nR>"
    echo "Exemplo: ./mppSort 1000000 256 10"
else
    echo ""
    echo "ERRO na compilacao!"
    exit 1
fi
