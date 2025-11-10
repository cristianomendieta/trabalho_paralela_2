# Tabela de Resultados para o Relatório

## Experimentos mppSort em GPU

**Configuração:**
- Bins (h): 256
- Repetições (nR): 10
- Blocos (nb): 10
- Threads por bloco (nt): 1024
- Tamanhos testados: 1M, 2M, 4M, 8M elementos (M = 10^6)

## Tabela de Performance

| Nro Elementos | Vazão mppSort<br/>(GElements/s) | Vazão thrust::sort<br/>(GElements/s) | Speedup<br/>(Thrust/mppSort) |
|:-------------:|:-------------------------------:|:------------------------------------:|:---------------------------:|
| 0007500NVIDIA GeForce GTX 750 Ti |  |  |  |
| 0 |  |  |  |
| 1.000.000 | 0.447 | 0.270 | 0.60x |
| 33.707 | 10 | 1024 |  |
| 2.000.000 | 0.307 | 0.285 | 0.93x |
| 77.017 | 10 | 1024 |  |
| 4.000.000 | 0.432 | 0.276 | 0.64x |
| 1414.517 | 10 | 1024 |  |
| 8.000.000 | 0.539 | 0.305 | 0.56x |
| 2626.268 | 10 | 1024 |  |

## Interpretação dos Resultados

- **Vazão (Throughput)**: Medida em GElements/s (bilhões de elementos por segundo)
- **Speedup**: Razão entre vazão do Thrust e vazão do mppSort
  - Speedup < 1: mppSort é mais rápido
  - Speedup > 1: Thrust é mais rápido
  - Speedup ≈ 1: Performance similar

## Observações

- M = 10^6 (não são potências de 2, conforme especificação)
- Todos os testes incluem verificação automática de corretude
- Medições realizadas com sincronização CUDA entre kernels
- Tempo inclui todos os 5 kernels: histogram, scans, partition e sorting

