if [ "$(hostname)" = "orval" ]; then
   echo "Compilacao especial na maquina orval"

   #compilação específica para GTX 750ti (máquina orval)
   #OBS:
   # nesse semestre a orval está com cuda 11.8 
   # nessa versao o nvcc NAO suporta gcc 12 ou g++ 12, 
   #   que é o gcc/g++ atualmente na orval
   # entao, apesar disso, consegui compilar com o gcc 12
   #   forçando o uso do gcc-12 conforme abaixo
   echo nvcc -arch sm_50 --allow-unsupported-compiler -ccbin /usr/bin/g++-12 thrust-sort.cu -o thrust-sort
   nvcc -arch sm_50 --allow-unsupported-compiler -ccbin /usr/bin/g++-12 thrust-sort.cu -o thrust-sort

elif [ "$(hostname)" = "nv00" ]; then

   echo "Compilacao especial na maquina nv00"
   echo "----- compilando especificamente para a GTX 1080ti  (sm_61)"
   echo "nvcc -arch sm_61 -I /usr/include/c++/10 -I /usr/lib/cuda/include/ thrust-sort.cu -o thrust-sort"
   nvcc -arch sm_61 -I /usr/include/c++/10 -I /usr/lib/cuda/include/ thrust-sort.cu -o thrust-sort
   #nvcc -arch sm_61 -o thrust-sort thrust-sort.cu    ## OU ISSO!

else

   #OBS para compilar para qualquer GPU basta retirar o -arch sm_61
   #    mas isso pode deixar a compilacao (ou a carga do programa) mais lenta

   echo compilando para maquina maquina genérica \(`hostname`\)
   #compilação para diversas GPUs
   echo nvcc -O3 thrust-sort.cu -o thrust-sort
   nvcc -O3 thrust-sort.cu -o thrust-sort

fi
