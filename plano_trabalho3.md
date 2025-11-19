## Plano de Trabalho: CI1009 - Trab 3 (mppSort GPU Otimizado)

**Objetivo Primário:** Implementar um algoritmo `mppSort` em CUDA (Parte B), utilizando um kernel de ordenação segmentada (`segmentedBitonicSort`) desenvolvido e testado separadamente (Parte A). O desempenho final será comparado ao `thrust::sort`.

**Dependências de Arquivo:**

  * `especificacao_trabalho3.txt` (Fornecida)
  * `especificacao_trabalho2.txt` (Fornecida)

-----

### Fase 1: Implementação (Parte A) - Testbed do Kernel `segmentedBitonicSort`

**Objetivo:** Criar um programa de teste autônomo (`segmented-sort-bitonic.c`) para desenvolver, testar e avaliar o kernel `segmentedBitonicSort` contra o `thrust::sort`.

  * **Tarefa 1.1: Criar o Programa Host (`segmented-sort-bitonic.c`)**

      * Implementar a lógica `main` em C/CUDA.
      * Implementar o *parser* de linha de comando para aceitar os dois formatos:
        1.  `Usage: segmented-sort-bitonic -n <total_elements> -segRange <min_seg> <max_seg>`
        2.  `Usage: segmented-sort-bitonic -n <total_elements> -ns <numberOfSegments>`

  * **Tarefa 1.2: Implementar Geração de Segmentos (Host)**

      * Com base nos argumentos da CLI, gerar aleatoriamente `num_segments` (seja de `-ns` ou do range).
      * Criar dois vetores no host (e copiá-los para a GPU):
          * `h_Offsets`: Posição inicial de cada segmento.
          * `h_Sizes`: Tamanho de cada segmento.
      * *Nota: A soma de `h_Sizes` deve ser igual a `total_elements`.*
      * Inicializar o vetor de chaves `d_Key` com `total_elements`.

  * **Tarefa 1.3: Implementar o Kernel `segmentedBitonicSort`**

      * Implementar o kernel `__global__` com a interface estrita:
        ```c++
        __global__ void segmentedBitonicSortKernel(
            uint *d_Key,
            uint *d_Offsets,
            uint *d_Sizes,
            uint num_segments,
            uint dir 
        );
        ```
      * **Lógica:** O kernel deve ser lançado com `num_segments` blocos (ou um grid que cubra `num_segments`).
      * Cada bloco `blockIdx.x` (ou thread) é responsável pelo `segment_id = blockIdx.x`.
      * O bloco lê seu offset (`d_Offsets[segment_id]`) e tamanho (`d_Sizes[segment_id]`).
      * O bloco carrega os dados (`d_Key + offset`) para a memória compartilhada (`__shared__`).
      * Executa um `bitonicSort` (similar ao do Trab 2 / CUDA Samples) dentro da memória compartilhada.
      * Escreve os dados ordenados de volta para a memória global em `d_Key + offset`.
      * *Nota: Este kernel deve ser otimizado, mas pode assumir que os segmentos cabem na memória compartilhada, conforme a premissa do `blockBitonicSort` do Trab 2.*

  * **Tarefa 1.4: Implementar Medição e Verificação**

      * Medir o tempo (usando `cudaEvent_t`) para:
        1.  `segmentedBitonicSort`: Execução do seu kernel.
        2.  `thrust::sort`: Execução do `thrust::sort` sobre *todo* o vetor `d_Key` (de tamanho `total_elements`).
      * Implementar a verificação de correção.
      * Imprimir a saída formatada (Tempo, Vazão em GEls/s, Speedup) conforme os exemplos na `especificacao_trabalho3.txt`.

-----

### Fase 2: Implementação (Parte B) - `mppSort` (Base Trab 2)

**Objetivo:** Construir o `mppSort` completo, implementando os kernels de particionamento do Trab 2 e corrigindo quaisquer "erros". Esta versão *não* usará o kernel da Parte A ainda.

  * **Tarefa 2.1: Criar o Programa Host (`mppSort.cu`)**

      * Implementar a lógica `main` conforme especificado no Trab 2.
      * Implementar o *parser* de CLI: `usage: ./mppSort <nTotalElements> h nR`
      * Implementar a geração de dados `Input` na CPU e a determinação de `nMin`, `nMax` na CPU, conforme permitido.

  * **Tarefa 2.2: Implementar Kernel 1 (`blockAndGlobalHisto`)**

      * Interface: `<<nb,nt>>( HH, Hg, h, Input, nElements, nMin, nMax );`
      * Implementar usando `__shared__` memória para o histograma local do bloco e `atomicAdd()` na memória global (`Hg`) no final.
      * `nt=1024`.

  * **Tarefa 2.3: Implementar Kernel 2 (`globalHistoScan`)**

      * Interface: `<<1,nt>>( Hg, SHg, h );`
      * Implementar um scan (soma de prefixos exclusiva) em `__shared__` memória.
      * `nt=1024`.

  * **Tarefa 2.4: Implementar Kernel 3 (`verticalScanHH`)**

      * Interface: `<<nb3,nt3>>( HH, PSv, h );` (Nota: A especificação tem um *typo* `(Hg, PSv, h)`, o correto é `(HH, PSv, h)` como entrada).
      * Implementar um scan vertical (por coluna) da matriz `HH` para `PSv`.

  * **Tarefa 2.5: Implementar Kernel 4 (`PartitionKernel`)**

      * Interface: `<<nb,nt>>( HH, SHg, PSv, h, Input, Output, nElements, nMin, nMax );`
      * Implementar a lógica de particionamento *exatamente* como descrito na seção "COMO implementar esse kernel" (v1.1) do Trab 2:
        1.  Carregar `PSv[b]` (provavelmente `HH[b]`, precisa verificar a lógica, a especificação está confusa. Vamos seguir a descrição literal "COMO implementar": "busca o histograma da linha b de **PSv**"). *Correção:* A descrição "COMO implementar" parece ter um *typo*. A lógica mais provável é:
            1.  Bloco `b` carrega `HH[b]` (seu histograma local) para `HLsh` (shared).
            2.  Threads calculam `HLsh[c] = SHg[c] + PSv[b*h + c]` (posição inicial global + offset vertical).
            3.  `__syncthreads()`.
            4.  Threads leem `Input`, calculam faixa `f`.
            5.  Usam `atomicAdd(&HLsh[f], 1)` para obter a posição `p`.
            6.  `Output[p] = e`.
                *Esta tarefa é a mais complexa e propensa a "erros". A implementação deve ser validada cuidadosamente.*

  * **Tarefa 2.6: Implementar Etapa de Ordenação (Estilo Trab 2)**

      * Implementar o **Kernel 5 (`blockBitonicSort`)** (importado do CUDA Samples).
      * Adicionar lógica host para iterar sobre as `h` faixas.
      * Para cada faixa `i`, verificar seu tamanho (`Hg[i]`).
      * Se `Hg[i]` cabe na shared memory: lançar o Kernel 5 (`blockBitonicSort`).
      * Se `Hg[i]` *não* cabe: lançar `thrust::sort` *inplace* no segmento `d_Output + SHg[i]`.

  * **Tarefa 2.7: Implementar Verificação**

      * Implementar a função `verifySort` (CPU ou GPU) que compara `Output` com um `thrust::sort` completo do `Input` original.

-----

### Fase 3: Refatoração (Parte B) - Integrando o Kernel da Parte A

**Objetivo:** Modificar o `mppSort` da Fase 2 para usar o kernel `segmentedBitonicSort` otimizado da Fase 1, eliminando a lógica complexa da Tarefa 2.6.

  * **Tarefa 3.1: Importar o Kernel**

      * Copiar o `segmentedBitonicSortKernel` (e funções `__device__` auxiliares) da Parte A para o `mppSort.cu` da Parte B.

  * **Tarefa 3.2: Substituir a Etapa de Ordenação**

      * Remover *completamente* a lógica da Tarefa 2.6 (Kernel 5, loops `thrust::sort`, verificação de tamanho de faixa).
      * Após a conclusão bem-sucedida do Kernel 4 (`PartitionKernel`), os vetores `d_Output`, `d_Hg` (tamanhos) e `d_SHg` (offsets) estão prontos.
      * Adicionar uma *única chamada* de kernel:
        ```c++
        // Lança o kernel da Parte A para ordenar todas as faixas de 'd_Output'
        // d_Output = d_Key
        // d_SHg    = d_Offsets
        // d_Hg     = d_Sizes
        // h        = num_segments
        segmentedBitonicSortKernel<<<grid, block>>>(
            d_Output, 
            d_SHg, 
            d_Hg, 
            h, 
            0 // dir
        );
        ```

  * **Tarefa 3.3: Medir Tempo Total**

      * Ajustar a medição de tempo para `nR` repetições.
      * O tempo total do `mppSort` (Parte B) é a soma dos tempos (K1 + K2 + K3 + K4 + `segmentedBitonicSortKernel`).

-----

### Fase 4: Experimentação e Relatório Final

**Objetivo:** Coletar dados de desempenho e gerar o relatório em PDF.

  * **Tarefa 4.1: Executar Benchmarks**

      * Executar o programa `mppSort.cu` (da Fase 3) para `nTotalElements`: 1M, 2M, 4M, 8M (M = 10^6).
      * Executar um benchmark separado (`thrust::sort` sozinho) para os mesmos `nTotalElements`.
      * Registrar os tempos de execução.

  * **Tarefa 4.2: Gerar Tabela de Resultados**

      * Criar uma tabela no relatório (sem gráficos) com as seguintes colunas:
        1.  `Tamanho da Entrada (N)`
        2.  `Tempo mppSort (s)`
        3.  `Tempo thrust::sort (s)`
        4.  `Vazão mppSort (GElementos/s)` (Cálculo: N / Tempo `mppSort` / 1e9)
        5.  `Vazão thrust::sort (GElementos/s)` (Cálculo: N / Tempo `thrust::sort` / 1e9)
        6.  `Aceleração (thrust / mppSort)`

  * **Tarefa 4.3: Escrever Relatório (PDF)**

      * Descrever a implementação (Parte A e B).
      * Explicar o funcionamento dos kernels (especialmente K1-K4 e o `PartitionKernel`).
      * Descrever a otimização (substituição do K5+Thrust pelo `segmentedBitonicSort`).
      * Incluir a Tabela de Resultados (Tarefa 4.2).

-----

### Fase 5: Entrega

  * **Tarefa 5.1: Agrupar Artefatos**
      * `relatorio.pdf`
      * Código-fonte da Parte A (`segmented-sort-bitonic.c` / `.cu`)
      * Código-fonte da Parte B (`mppSort.cu`)
      * (Opcional) Makefile para compilação.