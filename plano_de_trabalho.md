Este guia segue a estrutura definida no enunciado, dividindo o problema nos kernels especificados e nos requisitos de implementação.

-----

## Guia de Implementação: mppSort em GPU (CUDA)

### Passo 0: Configuração do Projeto, `main` e Gerenciamento de Dados

Antes de escrever qualquer kernel, configure seu arquivo `main.cu`, o tratamento de argumentos, a alocação de memória e a geração de dados.

1.  **Argumentos de Linha de Comando:**

      * Implemente a leitura dos argumentos: `./mppSort <nTotalElements> h <nR>`.
      * Armazene-os em variáveis (ex: `nTotalElements`, `h`, `nR`).

2.  **Geração de Dados (Host - CPU):**

      * Use *exatamente* o código fornecido no enunciado para gerar o vetor de entrada no host:
        ```cpp
        #include <stdlib.h> // Para rand()
        ...
        unsigned int *Input_host = (unsigned int*)malloc(nTotalElements * sizeof(unsigned int));
        unsigned int nMin = 0xFFFFFFFF;
        unsigned int nMax = 0;

        for( int i = 0; i < nTotalElements; i++ ){
            int a = rand();
            int b = rand();
            unsigned int v = a * 100 + b;
            Input_host[i] = v;

            // Encontrar Min e Max (conforme solicitado no enunciado)
            if (v < nMin) nMin = v;
            if (v > nMax) nMax = v;
        }

        // Imprimir Min, Max e a largura da faixa (L)
        unsigned int L = (nMax - nMin) / h;
        printf("Intervalo de dados [nMin, nMax]: [%u, %u]\n", nMin, nMax);
        printf("Largura da faixa (L): %u\n", L);
        ```
      * **Nota:** O enunciado permite que `nMin` e `nMax` sejam calculados na CPU, pois ela gerou os números. Isso simplifica muito os kernels.

3.  **Alocação de Memória (Device - GPU):**

      * Aloque toda a memória necessária na GPU.
      * `Input_dev`: `cudaMalloc(&Input_dev, nTotalElements * sizeof(unsigned int));`
      * `Output_dev`: `cudaMalloc(&Output_dev, nTotalElements * sizeof(unsigned int));`
      * `nb` (número de blocos): Defina `nb = NP * 2` (conforme enunciado). Um valor razoável para `NP` (Número de Streaming Multiprocessors) pode ser obtido via `cudaGetDeviceProperties`, ou um valor fixo como 64 ou 128 pode ser usado. Vamos usar `nb = 128` como exemplo.
      * `nt` (threads por bloco): `nt = 1024` (conforme enunciado).
      * `HH_dev`: `cudaMalloc(&HH_dev, nb * h * sizeof(unsigned int));` (Matriz `nb` x `h`)
      * `Hg_dev`: `cudaMalloc(&Hg_dev, h * sizeof(unsigned int));` (Vetor global)
      * `SHg_dev`: `cudaMalloc(&SHg_dev, h * sizeof(unsigned int));` (Scan do global)
      * `PSv_dev`: `cudaMalloc(&PSv_dev, nb * h * sizeof(unsigned int));` (Prefix Sum Vertical)
      * **Importante:** Zere os vetores de histograma/scan antes de usá-los: `cudaMemset(Hg_dev, 0, h * sizeof(unsigned int));`, `cudaMemset(HH_dev, 0, nb * h * sizeof(unsigned int));`, etc.

4.  **Transferência de Dados:**

      * Copie o vetor de entrada do host para o device:
        `cudaMemcpy(Input_dev, Input_host, nTotalElements * sizeof(unsigned int), cudaMemcpyHostToDevice);`

-----

### Passo 1: Kernel 1 - `blockAndGlobalHisto`

Este kernel lê a entrada e calcula *simultaneamente* os histogramas por bloco (`HH`) e o histograma global (`Hg`).

  * **Interface:** `blockAndGlobalHisto<<<nb, nt>>>(HH_dev, Hg_dev, h, Input_dev, nElements, nMin, nMax);`
  * **Lógica:**
    ```cuda
    __global__ void blockAndGlobalHisto(unsigned int* HH, unsigned int* Hg, int h, 
                                        unsigned int* Input, int nElements, 
                                        unsigned int nMin, unsigned int nMax)
    {
        // 1. Alocar histograma local em shared memory (obrigatório)
        // Assume-se que 'h' cabe em shared memory.
        extern __shared__ unsigned int s_histo[]; // nt*sizeof(int)

        // 2. Calcular largura da faixa (L)
        // (Cuidado com divisão por zero se nMax == nMin, mas para dados aleatórios é raro)
        unsigned int L = (nMax - nMin) / h;
        if (L == 0) L = 1; // Evitar divisão por zero

        // 3. Inicializar shared memory
        if (threadIdx.x < h) {
            s_histo[threadIdx.x] = 0;
        }
        __syncthreads();

        // 4. Loop grid-stride para processar todos os elementos
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = idx; i < nElements; i += stride) {
            unsigned int val = Input[i];
            
            // 5. Encontrar a faixa (bin) para o valor
            unsigned int bin = (val - nMin) / L;
            if (bin >= h) bin = h - 1; // Prender ao último bin

            // 6. Incrementar o histograma local em shared memory
            atomicAdd(&s_histo[bin], 1);
        }
        __syncthreads();

        // 7. Escrever resultados para a memória global
        // Cada thread escreve um bin do histograma local
        if (threadIdx.x < h) {
            unsigned int count = s_histo[threadIdx.x];
            
            // 8. Escrever no HH (matriz de histograma por bloco)
            HH[blockIdx.x * h + threadIdx.x] = count;
            
            // 9. Adicionar ao Hg (histograma global)
            atomicAdd(&Hg[threadIdx.x], count);
        }
    }
    ```
  * [cite\_start]**Referência:** Corresponde ao "Kernel 1: BlockAndGlobal Histo"[cite: 515, 591].

-----

### Passo 2: Kernel 2 - `globalHistoScan`

Este kernel calcula a **soma de prefixos exclusiva (Scan)** do histograma global `Hg`.

  * **Interface:** `globalHistoScan<<<1, 1024>>>(Hg_dev, SHg_dev, h);`
  * **Lógica:**
      * Este é um algoritmo de *Scan* paralelo padrão executado por um único bloco, usando shared memory (obrigatório pelo enunciado).
      * **Fase 1 (Up-Sweep / Reduce):** Carregue `Hg` para shared memory. Construa uma árvore de soma (redução) dentro da shared memory.
      * **Fase 2 (Down-Sweep / Scan):** Percorra a árvore para baixo, distribuindo as somas parciais para calcular o scan exclusivo. O primeiro elemento deve ser 0.
      * **Resultado:** Escreva o resultado de shared memory para `SHg_dev`. `SHg[i]` conterá a posição de início da faixa `i` no vetor de saída.
  * [cite\_start]**Referência:** Corresponde ao "Kernel 2: GlobalHistoScan"[cite: 560, 620].

-----

### Passo 3: Kernel 3 - `verticalScanHH`

Este kernel calcula a **soma de prefixos exclusiva (Scan)** *por coluna* da matriz `HH`.

  * **Interface:** `verticalScanHH<<<nb3, nt3>>>(HH_dev, PSv_dev, h, nb);`
      * **Nota:** O `Hg` na interface do enunciado parece ser um erro; a entrada deve ser `HH`. O `nb` (número de linhas/blocos de HH) é necessário.
  * **Estratégia de Lançamento:** A forma mais simples de paralelizar isso é lançar `h` blocos (um por coluna).
      * `nb3 = h`
      * `nt3 = 1024` (ou menos, se `nb` for pequeno)
  * **Lógica (Kernel com 1 bloco por coluna):**
    ```cuda
    __global__ void verticalScanHH(unsigned int* HH, unsigned int* PSv, int h, int nb)
    {
        int col = blockIdx.x; // Cada bloco cuida de uma coluna 'col'
        if (col >= h) return;

        // 1. Alocar shared memory para a coluna
        // (Assume-se 'nb' pequeno o suficiente, ex: 128)
        extern __shared__ unsigned int s_col[]; 

        // 2. Carregar coluna de HH para shared memory
        if (threadIdx.x < nb) {
            s_col[threadIdx.x] = HH[threadIdx.x * h + col];
        } else {
            // Preencher o resto para o algoritmo de scan (se necessário)
        }
        __syncthreads();

        // 3. Executar Scan Exclusivo em s_col
        //    (Use um algoritmo de scan padrão em shared memory)
        //    ... (implementação do scan) ...
        //    O resultado do scan deve estar em s_col

        // 4. Escrever o resultado (PSv)
        if (threadIdx.x < nb) {
            PSv[threadIdx.x * h + col] = s_col[threadIdx.x];
        }
    }
    ```
  * [cite\_start]**Referência:** Corresponde ao "Kernel 3: verticalScanHH"[cite: 563].

-----

### Passo 4: Kernel 4 - `PartitionKernel`

Este é o kernel mais complexo. Ele usa `SHg` e `PSv` para mover os dados de `Input` para `Output`, colocando-os na "faixa" correta (mas ainda não ordenados dentro da faixa).

  * **Interface:** `PartitionKernel<<<nb, nt>>>(HH_dev, SHg_dev, PSv_dev, h, Input_dev, Output_dev, nElements, nMin, nMax, nb);`
      * **Nota:** O enunciado v1.1 indica que `HH` é necessário. `nb` também é necessário.
  * **Lógica:** Segue *exatamente* a descrição "COMO implementar esse kernel" do enunciado (v1.1).
    ```cuda
    __global__ void PartitionKernel(unsigned int* HH, unsigned int* SHg, unsigned int* PSv, 
                                    int h, unsigned int* Input, unsigned int* Output, 
                                    int nElements, unsigned int nMin, unsigned int nMax, int nb)
    {
        int b = blockIdx.x; // ID do Bloco

        // 1. Alocar vetores em shared memory (obrigatório)
        extern __shared__ unsigned int s_data[];
        unsigned int* HLsh = s_data; // Tamanho 'h'
        unsigned int* SHg_sh = (unsigned int*)&s_data[h]; // Tamanho 'h'
        // (Certifique-se de alocar 2*h*sizeof(int) de shared memory no lançamento)

        // 2. Calcular largura da faixa (L)
        unsigned int L = (nMax - nMin) / h;
        if (L == 0) L = 1;

        // 3. Carregar SHg e PSv[b] para shared memory
        if (threadIdx.x < h) {
            HLsh[threadIdx.x] = PSv[b * h + threadIdx.x];
            SHg_sh[threadIdx.x] = SHg[threadIdx.x];
        }
        __syncthreads();

        // 4. Calcular a posição inicial (HLsh = PSv[b] + SHg)
        if (threadIdx.x < h) {
            HLsh[threadIdx.x] = HLsh[threadIdx.x] + SHg_sh[threadIdx.x];
        }
        __syncthreads();

        // 5. Loop grid-stride para processar os elementos do bloco
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = idx; i < nElements; i += stride) {
            unsigned int e = Input[i];
            
            // 6. Encontrar a faixa 'f'
            unsigned int f = (e - nMin) / L;
            if (f >= h) f = h - 1;
            
            // 7. Obter a posição de escrita com atomicAdd em shared memory
            unsigned int p = atomicAdd(&HLsh[f], 1);
            
            // 8. Armazenar o elemento na posição correta em Output
            Output[p] = e;
        }
    }
    ```
  * [cite\_start]**Referência:** Corresponde ao "Kernel 4: PartitionKernel"[cite: 553].

-----

### Passo 5: Kernel 5 - `blockBitonicSort` (e Thrust)

Após o Passo 4, `Output_dev` está particionado por faixa. Agora, você precisa ordenar *dentro* de cada faixa.

  * **Lógica (Host-side):**
    1.  Você precisa de `Hg_dev` (que contém o *tamanho* de cada faixa) e `SHg_dev` (que contém o *início* de cada faixa).
    2.  Copie `Hg_dev` e `SHg_dev` de volta para o Host (CPU).
        ```cpp
        unsigned int *Hg_host = (unsigned int*)malloc(h * sizeof(unsigned int));
        unsigned int *SHg_host = (unsigned int*)malloc(h * sizeof(unsigned int));
        cudaMemcpy(Hg_host, Hg_dev, h * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(SHg_host, SHg_dev, h * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        ```
    3.  Itere pelas faixas na CPU e lance kernels de ordenação:
        ```cpp
        #include <thrust/device_ptr.h>
        #include <thrust/sort.h>

        // Importe o kernel 'bitonicSort' dos samples da NVIDIA
        // Ex: __global__ void bitonicSort(unsigned int *data, ...)

        for (int i = 0; i < h; i++) {
            unsigned int faixa_start_index = SHg_host[i];
            unsigned int faixa_count = Hg_host[i];

            if (faixa_count == 0) continue; // Faixa vazia

            // Apontar para o início da faixa na GPU
            unsigned int* faixa_ptr = Output_dev + faixa_start_index;

            // Condições para usar Bitonic Sort (ex: potência de 2 e < 48KB)
            bool isPowerOfTwo = (faixa_count > 0) && ((faixa_count & (faixa_count - 1)) == 0);
            bool fitsInShared = (faixa_count * sizeof(unsigned int)) <= (48 * 1024);

            if (isPowerOfTwo && fitsInShared) {
                // Lançar seu kernel bitonicSort
                // A configuração de lançamento (blocos/threads) depende
                // da implementação do bitonic sort que você pegou.
                // Ex: bitonicSort<<<1, faixa_count / 2>>>(faixa_ptr, ...);
            } else {
                // Usar Thrust para ordenar in-place (conforme enunciado)
                thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(faixa_ptr);
                thrust::sort(thrust_ptr, thrust_ptr + faixa_count);
            }
        }
        ```

-----

### Passo 6: Verificação e Relatório

1.  **Função `verifySort` (Host):**

      * Crie uma cópia do `Input_host` (ex: `Input_verify`).
      * Ordene `Input_verify` usando `std::sort` (CPU) ou crie um `Verify_dev` e use `thrust::sort` (GPU) nele.
      * Copie seu `Output_dev` final de volta para um `Output_host`.
      * Compare `Output_host` com `Input_verify` elemento por elemento.
      * Imprima "Ordenação correta\!" ou "ERRO NA ORDENAÇÃO".

2.  **Medição de Desempenho (Benchmarking):**

      * Use `cudaEvent_t` para medir o tempo.
      * Crie um loop que executa `nR` vezes.
      * **Medição mppSort:**
          * `cudaEventRecord(start);`
          * Loop `nR`: [Kernel 1, Kernel 2, Kernel 3, Kernel 4, Loop do Kernel 5]
          * `cudaEventRecord(stop);`
          * Calcule o tempo médio.
      * **Medição Thrust (para comparação):**
          * `cudaEventRecord(start_thrust);`
          * Loop `nR`: [Copiar `Input_host` para `Thrust_Input_dev`, `thrust::sort(Thrust_Input_dev)`]
          * `cudaEventRecord(stop_thrust);`
          * Calcule o tempo médio do thrust.
      * **Vazão:** `GElementos/s = (nTotalElements / (tempo_medio_segundos)) / 1e9;`

3.  **Relatório (PDF):**

      * Crie a tabela solicitada com os dados de 1M, 2M, 4M e 8M de elementos (M=10^6).
      * Colunas: | Nro Elementos | Vazão mppSort (GElementos/s) | Vazão thrust::sort (GElementos/s) | Aceleração (mppSort/thrust) |
      * Descreva sua implementação e quaisquer desafios encontrados.