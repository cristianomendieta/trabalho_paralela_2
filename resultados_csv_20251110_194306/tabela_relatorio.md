# Resultados Experimentais - mppSort GPU

**Data:** seg 10 nov 2025 19:43:22 -03
**GPU:** NVIDIA GeForce GTX 750 Ti
NVIDIA GeForce GTX 750 Ti
**Parâmetros:** h=256 bins, nR=10 repetições, nb=10 blocos, nt=1024 threads/bloco

## Tabela de Performance

| Nro Elementos | Vazão mppSort (GElements/s) | Vazão thrust::sort (GElements/s) | Aceleração |
|---------------|----------------------------|----------------------------------|-----------|
| 1.000.000 | 0.445 | 0.255 | 0.57x |
| 2.000.000 | 0.305 | 0.285 | 0.93x |
| 4.000.000 | 0.442 | 0.280 | 0.63x |
| 8.000.000 | 0.540 | 0.303 | 0.56x |

## Observações

- M = 10^6 (um milhão)
- Aceleração = Vazão Thrust / Vazão mppSort
  - Valores < 1: mppSort é mais rápido que Thrust
  - Valores > 1: Thrust é mais rápido que mppSort
- Verificação de corretude realizada automaticamente
