# Informações de Configuração nos Resultados

## 📊 O que foi adicionado

Seguindo o exemplo da planilha fornecida, agora os scripts incluem informações detalhadas da configuração de execução:

### Parâmetros Incluídos:

1. **h** - Número de bins do histograma (256)
2. **nR** - Número de repetições para timing (10)
3. **nb** - Número de blocos CUDA (NP * 2)
4. **nt** - Número de threads por bloco (1024)

## 📝 Formato do CSV Atualizado

### Antes:
```csv
# Parametros: h=256 bins, nR=10 repeticoes

Nro_Elementos,Vazao_mppSort_GElements_s,Vazao_Thrust_GElements_s,Speedup
```

### Agora:
```csv
# Parametros: h=256 bins, nR=10 repeticoes, nb=10 blocos, nt=1024 threads/bloco

Nro_Elementos,Vazao_mppSort_GElements_s,Vazao_Thrust_GElements_s,Speedup,Tempo_Medio_ms,Blocos,Threads_Bloco
```

## 🎯 Por que isso é importante?

### 1. **Reprodutibilidade**
Permite que outros reproduzam exatamente os mesmos experimentos com a mesma configuração.

### 2. **Documentação Completa**
O relatório terá todas as informações técnicas necessárias:
- Configuração da GPU
- Número de SMs (Streaming Multiprocessors)
- Configuração de blocos e threads
- Parâmetros do algoritmo (bins, repetições)

### 3. **Análise de Performance**
Com `nb` e `nt`, é possível:
- Entender a ocupação da GPU
- Calcular o número total de threads: `nb * nt`
- Analisar a granularidade da paralelização

## 📐 Exemplo de Cálculos

Para GPU GTX 750 Ti (5 SMs):
- **nb** = NP * 2 = 5 * 2 = **10 blocos**
- **nt** = **1024 threads/bloco**
- **Total de threads** = 10 * 1024 = **10.240 threads**

## 📋 Formato Semelhante ao Exemplo

O exemplo da planilha mostra:
```csv
Executando,10,vezes,com,10000000,elementos,e,8,threads
```

Nosso formato agora:
```csv
Executando,10,vezes,com,1000000,elementos,256,bins,10,blocos,1024,threads
```

## 🔧 Como Funciona

Os scripts agora:

1. **Executam um teste rápido** (1000 elementos) para extrair a configuração:
```bash
./mppSort 1000 256 1 > temp.txt
```

2. **Extraem as informações** da saída do programa:
```bash
NB=$(grep "Number of blocks" temp.txt | awk '{print $5}')
NT=$(grep "Threads per block" temp.txt | awk '{print $5}')
```

3. **Incluem nos resultados**:
   - No cabeçalho CSV
   - Em cada linha de dados
   - Na tabela Markdown formatada

## 📊 Exemplo de Saída Completa

```csv
# Resultados Experimentais - mppSort GPU
# Data: 2025-11-10
# GPU: NVIDIA GeForce GTX 750 Ti
# Parametros: h=256 bins, nR=10 repeticoes, nb=10 blocos, nt=1024 threads/bloco

Nro_Elementos,Vazao_mppSort_GElements_s,Vazao_Thrust_GElements_s,Speedup,Tempo_Medio_ms,Blocos,Threads_Bloco
1000000,1.234,2.345,1.90,12.345,10,1024
2000000,1.456,2.567,1.76,23.456,10,1024
4000000,1.678,2.789,1.66,45.678,10,1024
8000000,1.890,3.012,1.59,89.012,10,1024
```

## 📖 No Relatório

Agora você pode incluir uma seção como:

### Configuração Experimental

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| GPU | GTX 750 Ti | Compute Capability 5.0 |
| SMs | 5 | Streaming Multiprocessors |
| Blocos (nb) | 10 | NP * 2 |
| Threads/Bloco (nt) | 1024 | Máximo para essa GPU |
| Total Threads | 10.240 | nb * nt |
| Bins (h) | 256 | Faixas do histograma |
| Repetições (nR) | 10 | Para média de tempo |

## ✅ Scripts Atualizados

Ambos os scripts foram atualizados:

1. ✅ `gera_resultados_finais.sh`
2. ✅ `gera_csv_experimentos.sh`

## 🚀 Como Usar

```bash
# Mesmo comando de antes
./gera_resultados_finais.sh

# Agora com informações completas nos resultados!
```

---

**Nota:** Esta informação adicional torna o relatório mais completo e profissional, seguindo boas práticas de documentação científica! 📚✨
