# Correção do Erro "Illegal Memory Access"

## Problema Identificado

Erro na linha 373: `an illegal memory access was encountered`

## Causas Prováveis

1. **Shared Memory Insuficiente**: Kernel 2 estava alocando shared memory para apenas h elementos, mas lançando com 1024 threads
2. **Acesso Fora dos Limites no Kernel 4**: O atomicAdd poderia gerar índices p >= nElements
3. **Configuração de Threads Inadequada**: Kernels 2 e 3 não estavam configurados adequadamente

## Correções Aplicadas

### 1. Ajuste de Threads e Shared Memory - Kernel 2
```cuda
// ANTES:
int sharedMem2 = h * sizeof(unsigned int);
globalHistoScan<<<1, nt, sharedMem2>>>(...)  // nt=1024, mas shared=h=256

// DEPOIS:
int nt2 = (h < 256) ? 256 : h;
int sharedMem2 = nt2 * sizeof(unsigned int);
globalHistoScan<<<1, nt2, sharedMem2>>>(...)
```

### 2. Ajuste de Threads e Shared Memory - Kernel 3
```cuda
// ANTES:
int nt3 = (nb > 1024) ? 1024 : nb;
int sharedMem3 = nb * sizeof(unsigned int);

// DEPOIS:
int nt3 = 32;
while (nt3 < nb && nt3 < 1024) nt3 *= 2;
int sharedMem3 = nt3 * sizeof(unsigned int);
```

### 3. Proteção Contra Acesso Fora dos Limites - Kernel 4
```cuda
// ANTES:
unsigned int p = atomicAdd(&HLsh[f], 1);
Output[p] = e;

// DEPOIS:
unsigned int p = atomicAdd(&HLsh[f], 1);
if (p < nElements) {
    Output[p] = e;
}
```

### 4. Mensagens de Debug Adicionadas
Agora o programa imprime quando cada kernel é lançado na primeira iteração, facilitando identificar onde ocorre o erro.

## Como Testar

### Teste Muito Pequeno (Debug):
```bash
chmod +x teste_debug.sh
./teste_debug.sh
```
Testa com apenas 1000 elementos e 16 bins.

### Teste Pequeno:
```bash
./teste_rapido.sh
```
Testa com 10k, 100k, 500k elementos.

### Testes Completos:
```bash
./roda_experimentos.sh
```
Executa os 4 experimentos (1M, 2M, 4M, 8M).

## Próximos Passos

1. Recompilar:
   ```bash
   ./compila.sh
   ```

2. Testar com debug:
   ```bash
   ./teste_debug.sh
   ```

3. Se funcionar, rodar experimentos:
   ```bash
   ./roda_experimentos.sh
   ```

## Observações Importantes

- A GPU GTX 750 Ti tem compute capability 5.0 e apenas 5 SMs
- Com 5 SMs e nb = NP * 2 = 10 blocos, temos poucos blocos
- A shared memory máxima por bloco é 48KB
- Para h=256: 2*h*sizeof(uint) = 2KB (OK)
- Para nt=1024: 1024*sizeof(uint) = 4KB (OK)

O problema estava na configuração inadequada de threads vs shared memory alocada.
