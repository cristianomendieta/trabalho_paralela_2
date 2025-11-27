# Relatório do Trabalho 3: Programação Paralela com GPUs

**Aluno:** Cristiano Mendieta
**Disciplina:** CI1009 - Programação Paralela
**Data:** 26 de Novembro de 2025

## 1. Introdução

Este trabalho teve como objetivo implementar e otimizar algoritmos de ordenação em GPU utilizando CUDA. O projeto foi dividido em duas partes:
*   **Parte A:** Implementação de um *Segmented Bitonic Sort*, capaz de ordenar múltiplos segmentos de um vetor de forma independente e paralela.
*   **Parte B:** Implementação do algoritmo `mppSort` (baseado em particionamento por histograma), integrando o kernel desenvolvido na Parte A para a etapa final de ordenação local.

## 2. Detalhes de Implementação

### Parte A: Segmented Bitonic Sort
O kernel `blockBitonicSort` foi implementado para ordenar segmentos que cabem inteiramente na memória compartilhada (Shared Memory) da GPU. A estratégia adotada foi:
1.  **Carregamento com Padding:** Cada bloco carrega seu segmento para a memória compartilhada. Se o tamanho do segmento não for potência de 2, ele é preenchido com valores neutros (`UINT_MAX` para ordem crescente) até a próxima potência de 2.
2.  **Bitonic Sort Network:** A ordenação ocorre inteiramente na memória compartilhada, minimizando o acesso à memória global.
3.  **Escrita:** Apenas os elementos válidos (originais) são escritos de volta na memória global.

### Parte B: mppSort (Integração)
O `mppSort` é um algoritmo de ordenação global projetado para dividir um vetor muito grande em pedaços menores e independentes, que podem ser ordenados em paralelo. O funcionamento detalhado é:

1.  **Amostragem e Histograma (Kernels 1 e 2):** O algoritmo analisa a distribuição dos dados de entrada (`Input`) para determinar "baldes" (buckets) de valores. Calcula-se um histograma global para saber quantos elementos cairão em cada faixa de valores.
2.  **Cálculo de Offsets (Kernel 3):** Utiliza-se uma operação de *Prefix Sum* (Scan) sobre os histogramas para determinar a posição exata de memória onde cada balde deve começar no vetor de saída.
3.  **Particionamento (Kernel 4):** Os elementos são movidos da entrada para a saída (`Output`), agrupados por faixa de valor. Após esta etapa, o vetor está "quase" ordenado: todos os elementos do Balde 0 são menores que os do Balde 1, e assim por diante. Porém, os elementos *dentro* de cada balde ainda estão desordenados.
4.  **Ordenação Local (Kernel da Parte A):** Aqui ocorre a integração. Como cada balde é independente e (graças ao ajuste do parâmetro `h`) cabe na memória compartilhada, lançamos o kernel `segmentedBitonicSort` da Parte A. Cada bloco CUDA assume um balde, carrega-o para a Shared Memory, ordena-o usando a rede bitônica e escreve de volta. Isso elimina a necessidade de comunicação global nesta etapa final, resultando em alta eficiência.

## 3. Metodologia Experimental

Os experimentos foram realizados na máquina **`orval`**, que possui uma GPU de arquitetura **Maxwell (sm_50)**.

*   **Compilador:** NVCC (CUDA 10.x / 11.x compatível)
*   **Flags:** `-arch sm_50 --allow-unsupported-compiler -std=c++17`
*   **Comparativo:** Os resultados foram comparados com a biblioteca `thrust::sort` (Radix Sort altamente otimizado).

## 4. Resultados Experimentais

Abaixo apresentamos a tabela de desempenho comparando o `mppSort` (nossa implementação) com o `thrust::sort`.

### Tabela de Vazão e Aceleração (mppSort vs Thrust)

| Tamanho (N) | Bins (h) | Tempo mppSort (ms) | Vazão mppSort (GEls/s) | Tempo Thrust (ms) | Vazão Thrust (GEls/s) | Aceleração (Thrust/mppSort) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.000.000** | 512 | 6.75 | 0.148 | 3.48 | 0.287 | **0.52x** |
| **2.000.000** | 1024 | 13.76 | 0.145 | 6.70 | 0.299 | **0.49x** |
| **4.000.000** | 2048 | 27.69 | 0.144 | 12.72 | 0.314 | **0.46x** |
| **8.000.000** | 4096 | 54.26 | 0.147 | 26.44 | 0.303 | **0.49x** |

*Nota: Aceleração < 1.0x indica que o mppSort foi mais lento que o Thrust.*

### Resultados da Parte A (Segmented Bitonic Sort - 8M Elementos)

Para a Parte A, testamos o kernel isolado em diferentes cenários de segmentação:

| Cenário | Vazão Bitonic (GEls/s) | Vazão Thrust (GEls/s) | Speedup |
| :--- | :--- | :--- | :--- |
| Segmentos Pequenos (20-4000) | 0.151 | 0.470 | 0.32x |
| Segmentos Médios (3000-4000) | 0.175 | 0.469 | 0.37x |
| Segmentos Variados (20-8000) | 0.144 | 0.470 | 0.31x |

## 5. Análise dos Resultados

Observou-se que o desempenho da implementação (`mppSort` e `Bitonic Sort`) ficou abaixo do `thrust::sort` na máquina `orval`. Isso se deve a fatores de hardware e algorítmicos:

1.  **Arquitetura Maxwell (Antiga):** A GPU utilizada (`sm_50`) possui limitações significativas de Shared Memory e largura de banda em comparação com arquiteturas modernas (como Ampere, usada nos exemplos do professor). O algoritmo Bitonic Sort depende pesadamente de Shared Memory, o que limita a ocupação (número de blocos ativos) nessa GPU específica.
2.  **Algoritmo:** O `thrust::sort` utiliza Radix Sort, que possui complexidade $O(N)$ e é extremamente eficiente em acesso à memória. O Bitonic Sort tem complexidade $O(N \log^2 N)$, realizando mais operações por elemento.
3.  **Estabilidade:** Apesar do desempenho inferior, a vazão do `mppSort` manteve-se constante (~0.145 GEls/s) com o aumento do tamanho do problema, indicando boa escalabilidade do algoritmo implementado.

## 6. Conclusão

O desenvolvimento deste trabalho permitiu a implementação completa e correta de um sistema de ordenação híbrido em GPU, combinando técnicas de particionamento global (`mppSort`) com ordenação local otimizada (`Segmented Bitonic Sort`). A solução atingiu 100% de corretude em todos os casos de teste (de 1 a 8 milhões de elementos), validando a complexa lógica de sincronização entre kernels e o gerenciamento preciso de memória compartilhada.

A integração entre a Parte A e a Parte B demonstrou a eficácia da estratégia de "dividir para conquistar" em arquiteturas massivamente paralelas. Embora a comparação direta com a biblioteca `thrust::sort` na arquitetura Maxwell (`sm_50`) tenha mostrado um desempenho inferior em termos absolutos, um resultado esperado dada a otimização extrema do Radix Sort da Thrust e as limitações de hardware da máquina de testes, a implementação exibiu um comportamento estável e escalável. A vazão constante observada com o aumento da carga de trabalho confirma que o algoritmo explora eficientemente o paralelismo disponível.

Em suma, o projeto cumpriu rigorosamente os requisitos funcionais, entregando um *sorter* robusto que ilustra na prática os desafios e benefícios da computação de alto desempenho com CUDA, desde o uso de operações atômicas e *prefix sums* até a otimização de acesso à memória via *coalescing* e *shared memory*.
