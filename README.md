# mppSort - GPU Parallel Sorting Implementation

**Trabalho 2 - CI1009 Programação Paralela com GPUs**  
**UFPR - Universidade Federal do Paraná**  
**Autores:** Cristiano Mendieta e Thiago Ruiz  
**Data:** Novembro de 2025

Implementação do algoritmo de ordenação paralela mppSort em GPU usando CUDA, conforme especificado no Trabalho 2 da disciplina (versão 1.1).

## Descrição

Este trabalho implementa uma versão paralela do algoritmo mppSort para GPUs utilizando CUDA. O algoritmo utiliza técnicas de:
- Histogramas paralelos
- Soma de prefixos (scan)
- Particionamento de dados
- Ordenação bitônica para bins pequenos
- Thrust para bins maiores

## Estrutura do Código

O algoritmo consiste em 5 kernels principais:

### Kernel 1: blockAndGlobalHisto
- Calcula histogramas por bloco (HH) e histograma global (Hg)
- Usa shared memory para otimização
- Cada bloco processa uma parte dos dados de entrada

### Kernel 2: globalHistoScan
- Calcula a soma de prefixos exclusiva do histograma global (Hg → SHg)
- Implementa o algoritmo de Blelloch scan
- Executa em um único bloco usando shared memory

### Kernel 3: verticalScanHH
- Calcula soma de prefixos por coluna da matriz HH
- Produz a matriz PSv (Prefix Sum Vertical)
- Lança h blocos (um por coluna)

### Kernel 4: PartitionKernel
- Particiona os dados de entrada no vetor de saída
- Usa SHg e PSv para determinar posições corretas
- Utiliza atômicos em shared memory para eficiência

### Kernel 5: Ordenação por Bins
- bitonicSort: ordena bins que são potência de 2 e cabem em shared memory (≤48KB)
- thrust::sort: ordena bins maiores ou que não são potência de 2

## Compilação

### Requisitos
- CUDA Toolkit instalado
- GPU com suporte CUDA (compute capability ≥ 6.0)
- Compilador nvcc

### Compilar o Código

```bash
chmod +x compila.sh
./compila.sh
```

O script de compilação detecta automaticamente se o CUDA está instalado e compila com otimizações.

**Nota:** Você pode precisar ajustar a flag `-arch` no script `compila.sh` de acordo com sua GPU:
- `-arch=sm_60`: Pascal (GTX 10xx)
- `-arch=sm_75`: Turing (RTX 20xx)
- `-arch=sm_80`: Ampere (A100)
- `-arch=sm_86`: Ampere (RTX 30xx)
- `-arch=sm_89`: Ada Lovelace (RTX 40xx)

## Uso

```bash
./mppSort <nTotalElements> <h> <nR>
```

### Parâmetros:
- `nTotalElements`: número de elementos inteiros sem sinal no vetor de entrada
- `h`: número de bins do histograma
- `nR`: número de repetições para medição de tempo

### Exemplos:

```bash
# 1 milhão de elementos, 256 bins, 10 repetições
./mppSort 1000000 256 10

# 2 milhões de elementos, 512 bins, 20 repetições
./mppSort 2000000 512 20

# 4 milhões de elementos, 1024 bins, 10 repetições
./mppSort 4000000 1024 10

# 8 milhões de elementos, 256 bins, 10 repetições
./mppSort 8000000 256 10
```

## Saída

O programa exibe:
1. **Informações do dispositivo:** GPU detectada, número de SMs, configuração de blocos/threads
2. **Intervalo de dados:** [nMin, nMax] e largura dos bins (L)
3. **Performance do mppSort:** tempo médio e vazão (GElements/s)
4. **Performance do Thrust:** tempo médio e vazão (GElements/s) para comparação
5. **Speedup:** aceleração do mppSort em relação ao Thrust
6. **Verificação:** confirmação se a ordenação está correta

## Estrutura de Arquivos

```
trabalho_2_paralela/
├── especificacao_trabalho.txt    # Especificação do trabalho
├── plano_de_trabalho.md          # Plano de implementação
├── mppSort.cu                     # Código fonte principal
├── compila.sh                     # Script de compilação
├── README.md                      # Este arquivo
└── thrust-sort/                   # Exemplo de referência
    ├── thrust-sort.cu
    └── chrono.c
```

## Experimentos

Para os experimentos do relatório, execute com os seguintes parâmetros:

```bash
# Experimento 1: 1M elementos
./mppSort 1000000 256 10

# Experimento 2: 2M elementos
./mppSort 2000000 256 10

# Experimento 3: 4M elementos
./mppSort 4000000 256 10

# Experimento 4: 8M elementos
./mppSort 8000000 256 10
```

**Nota:** M = 10^6 (não use potências de 2 para o número de elementos)

## Verificação de Corretude

O programa automaticamente verifica se a ordenação está correta comparando com uma ordenação feita com `std::sort` no host. A saída indicará:
- "Ordenação correta!" se o resultado está correto
- "ERRO NA ORDENAÇÃO" se houver discrepâncias

## Detalhes de Implementação

### Geração de Dados
Os dados de entrada são gerados conforme especificação:
```c
unsigned int v = rand() * 100 + rand();
```

### Configuração de Blocos e Threads
- `nb = NP * 2` (onde NP = número de multiprocessadores)
- `nt = 1024` threads por bloco

### Uso de Shared Memory
Todos os kernels críticos (1, 2, 3, 4) usam shared memory para otimização de performance.

### Atomics
- Kernel 1: atomicAdd em Hg (global)
- Kernel 4: atomicAdd em HLsh (shared memory)

## Referências

- Especificação do trabalho: `especificacao_trabalho.txt`
- Plano de trabalho: `plano_de_trabalho.md`
- Algoritmo mppSort original: Cordeiro, M. B.; Blanco, R. M.; Zola, W. M. N. "Algoritmo Paralelo Eficiente para Ordenação Chave-Valor". XL Simpósio Brasileiro de Bancos de Dados (SBBD 2025), 2025
- CUDA Samples: Bitonic Sort (`/usr/local/cuda/cuda-samples/Samples/2_Concepts_and_Techniques/sortingNetworks`)

## Autores

Cristiano Mendieta
CI1009 - Programação Paralela com GPUs
UFPR - 2o Semestre 2025

## Licença

Este código é parte de um trabalho acadêmico para a disciplina CI1009.
