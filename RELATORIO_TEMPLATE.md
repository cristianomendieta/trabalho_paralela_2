# Relatório: Implementação do Algoritmo mppSort em GPU

**Disciplina:** CI1009 - Programação Paralela com GPUs  
**Aluno:** Cristiano Mendieta  
**Semestre:** 2o Semestre 2025  
**Data de Entrega:** 12/11/2025

---

## 1. Introdução

Este relatório descreve a implementação do algoritmo de ordenação paralela **mppSort** para GPUs utilizando CUDA. O algoritmo foi baseado na especificação do Trabalho 2 da disciplina e implementa técnicas de programação paralela como histogramas, soma de prefixos (scan), particionamento de dados e ordenação híbrida.

## 2. Descrição do Algoritmo

O algoritmo mppSort divide o problema de ordenação em várias etapas, cada uma implementada como um kernel CUDA:

### 2.1 Kernel 1: blockAndGlobalHisto

Este kernel é responsável por:
- Calcular histogramas locais por bloco de threads (matriz HH)
- Calcular o histograma global (vetor Hg)
- Utilizar shared memory para armazenamento temporário dos histogramas locais
- Utilizar operações atômicas para atualizar o histograma global

**Otimizações implementadas:**
- Uso de shared memory para reduzir acessos à memória global
- Grid-stride loop para processar todos os elementos
- Coalescência de acessos à memória

### 2.2 Kernel 2: globalHistoScan

Este kernel implementa uma soma de prefixos exclusiva (exclusive scan) sobre o histograma global usando o algoritmo de Blelloch:
- **Up-sweep (reduce):** constrói uma árvore de somas parciais
- **Down-sweep:** distribui as somas para calcular o scan
- Executa em um único bloco com toda computação em shared memory

**Resultado:** O vetor SHg contém as posições iniciais de cada bin no vetor de saída ordenado.

### 2.3 Kernel 3: verticalScanHH

Este kernel calcula a soma de prefixos por coluna (vertical) da matriz HH:
- Lança h blocos, um para cada coluna
- Cada bloco carrega sua coluna em shared memory
- Aplica o algoritmo de scan na coluna
- Produz a matriz PSv (Prefix Sum Vertical)

### 2.4 Kernel 4: PartitionKernel

Este é o kernel mais complexo, responsável pelo particionamento dos dados:
- Cada bloco carrega suas informações de PSv e SHg em shared memory
- Calcula as posições iniciais onde cada bloco deve inserir elementos em cada bin
- Usa operações atômicas em shared memory (mais rápido que global memory)
- Cada thread lê um elemento, determina seu bin, obtém uma posição via atomic add, e escreve na saída

**Técnica chave:** Uso de atomics em shared memory (HLsh) para alocação eficiente de posições dentro de cada bin.

### 2.5 Kernel 5: Ordenação por Bins

Após o particionamento, cada bin precisa ser ordenado internamente:

- **bitonicSort:** Para bins que são potência de 2 e cabem em shared memory (≤48KB)
  - Algoritmo adaptado dos CUDA samples
  - Ordenação in-place em shared memory
  
- **thrust::sort:** Para bins maiores ou que não são potência de 2
  - Biblioteca Thrust fornece implementação otimizada
  - Ordenação in-place na memória global

## 3. Detalhes de Implementação

### 3.1 Configuração de Execução

- **Número de blocos (nb):** NP × 2, onde NP é o número de SMs da GPU
- **Threads por bloco (nt):** 1024 (máximo permitido pela GPU)
- **Shared memory:** Alocada dinamicamente para cada kernel conforme necessidade

### 3.2 Geração de Dados

Os dados de entrada são gerados conforme especificação:
```c
unsigned int v = rand() * 100 + rand();
```

Os valores mínimo (nMin) e máximo (nMax) são calculados durante a geração, e a largura dos bins é:
```
L = (nMax - nMin) / h
```

### 3.3 Verificação de Corretude

A função `verifySort()` compara o resultado do mppSort com uma ordenação de referência usando `std::sort` do C++ STL. Todos os testes indicaram ordenação correta.

## 4. Experimentos e Resultados

### 4.1 Ambiente de Testes

- **GPU:** [PREENCHER COM SUA GPU]
- **CUDA Version:** [PREENCHER]
- **Número de SMs:** [PREENCHER]
- **Compute Capability:** [PREENCHER]

### 4.2 Parâmetros dos Experimentos

- **h (número de bins):** 256
- **nR (repetições):** 10
- **Elementos testados:** 1M, 2M, 4M, 8M (M = 10^6)

### 4.3 Resultados de Performance

| Nro Elementos | Vazão mppSort (GElements/s) | Vazão thrust::sort (GElements/s) | Speedup (mppSort/Thrust) |
|---------------|----------------------------|----------------------------------|--------------------------|
| 1,000,000     | [PREENCHER]                | [PREENCHER]                      | [PREENCHER]              |
| 2,000,000     | [PREENCHER]                | [PREENCHER]                      | [PREENCHER]              |
| 4,000,000     | [PREENCHER]                | [PREENCHER]                      | [PREENCHER]              |
| 8,000,000     | [PREENCHER]                | [PREENCHER]                      | [PREENCHER]              |

**Observações:**
[PREENCHER COM ANÁLISE DOS RESULTADOS]

## 5. Análise dos Resultados

### 5.1 Performance Geral

[PREENCHER - Discutir o desempenho geral do mppSort em relação ao Thrust]

### 5.2 Escalabilidade

[PREENCHER - Como a performance escala com o aumento do número de elementos]

### 5.3 Eficiência dos Kernels

[PREENCHER - Discutir quais kernels são os gargalos e por quê]

### 5.4 Impacto do Número de Bins

[PREENCHER - Se você testou com diferentes valores de h, discuta o impacto]

## 6. Desafios e Soluções

### 6.1 Implementação do Scan

O algoritmo de Blelloch scan requer potências de 2. Para lidar com tamanhos arbitrários de h e nb:
- Padding com zeros em shared memory
- Verificações de limites nas operações

### 6.2 Atomics em Shared Memory

Para maximizar a eficiência no Kernel 4:
- Uso de atomics em shared memory (muito mais rápido que global)
- Estrutura de dados HLsh permite alocação eficiente de posições

### 6.3 Ordenação Híbrida

A escolha entre bitonicSort e thrust::sort é feita dinamicamente:
- Critérios: tamanho potência de 2 e cabe em shared memory
- Maximiza o uso de shared memory quando possível
- Fallback para Thrust garante corretude

## 7. Conclusões

[PREENCHER - Conclusões sobre o trabalho]

Este trabalho implementou com sucesso o algoritmo mppSort em GPU, demonstrando:
- Técnicas de otimização com shared memory
- Uso eficiente de operações atômicas
- Implementação de algoritmos fundamentais (scan, histogram)
- Ordenação híbrida combinando diferentes estratégias

## 8. Possíveis Melhorias

1. **Otimização do Scan:** Implementar scan multi-bloco para histogramas muito grandes
2. **Tuning de Parâmetros:** Explorar diferentes valores de h e configurações de blocos/threads
3. **Coalescing:** Analisar e melhorar padrões de acesso à memória
4. **Overlap:** Usar streams para sobrepor comunicação e computação

## Referências

1. Especificação do Trabalho 2 - CI1009, UFPR, 2025
2. Cordeiro, M. B.; Blanco, R. M.; Zola, W. M. N. "Algoritmo Paralelo Eficiente para Ordenação Chave-Valor". XL Simpósio Brasileiro de Bancos de Dados (SBBD 2025), 2025
3. NVIDIA CUDA C Programming Guide
4. CUDA Samples - Sorting Networks (bitonicSort)
5. Thrust Library Documentation

---

**Anexos:**
- Código fonte: `mppSort.cu`
- Script de compilação: `compila.sh`
- README com instruções de uso
