# Scripts para Geração de Resultados CSV

## 📊 Scripts Disponíveis

### 1. `gera_resultados_finais.sh` ⭐ **RECOMENDADO**

Script simplificado que gera os resultados no formato adequado para o relatório.

**Uso:**
```bash
./gera_resultados_finais.sh
```

**O que faz:**
- Executa os 4 experimentos obrigatórios (1M, 2M, 4M, 8M)
- Gera arquivo CSV com os dados
- Cria tabela formatada em Markdown pronta para o relatório
- Parâmetros: h=256 bins, nR=10 repetições

**Arquivos gerados:**
- `resultados_finais_*/resultados_mppsort.csv` - Dados brutos
- `resultados_finais_*/tabela_para_relatorio.md` - Tabela formatada

---

### 2. `gera_csv_experimentos.sh`

Script mais detalhado com múltiplos formatos de saída.

**Uso:**
```bash
./gera_csv_experimentos.sh
```

**O que faz:**
- Executa os 4 experimentos
- Gera 3 arquivos de saída:
  - CSV detalhado
  - CSV resumo
  - Tabela Markdown

---

### 3. `teste_debug.sh`

Teste minúsculo para debug (1000 elementos).

**Uso:**
```bash
./teste_debug.sh
```

---

### 4. `teste_rapido.sh`

Testes rápidos com 10k, 100k, 500k elementos.

**Uso:**
```bash
./teste_rapido.sh
```

---

### 5. `roda_experimentos.sh`

Script original que executa os experimentos completos.

**Uso:**
```bash
./roda_experimentos.sh
```

---

## 📋 Formato da Especificação

Conforme o arquivo `especificacao_trabalho.txt`, o relatório deve incluir:

### Tabela Requerida:

| Nro Elementos | Vazão mppSort (GElements/s) | Vazão thrust::sort (GElements/s) | Aceleração |
|---------------|----------------------------|----------------------------------|-----------|
| 1.000.000     | X.XXX                      | Y.YYY                            | Z.ZZx     |
| 2.000.000     | X.XXX                      | Y.YYY                            | Z.ZZx     |
| 4.000.000     | X.XXX                      | Y.YYY                            | Z.ZZx     |
| 8.000.000     | X.XXX                      | Y.YYY                            | Z.ZZx     |

**Requisitos:**
- ✅ Tamanhos: 1M, 2M, 4M, 8M elementos (M = 10^6, **NÃO potências de 2**)
- ✅ Vazão em **GElementos/s**
- ✅ Comparação com Thrust
- ✅ Aceleração (Speedup)
- ❌ Não precisa de gráficos

---

## 🎯 Workflow Recomendado

### Passo 1: Recompilar (se necessário)
```bash
./compila.sh
```

### Passo 2: Teste Rápido (validação)
```bash
./teste_debug.sh
```

### Passo 3: Gerar Resultados para o Relatório
```bash
./gera_resultados_finais.sh
```

### Passo 4: Copiar Dados para o Relatório
```bash
# Ver os resultados
cat resultados_finais_*/resultados_mppsort.csv

# Ver a tabela formatada
cat resultados_finais_*/tabela_para_relatorio.md
```

---

## 📝 Formato do CSV Gerado

```csv
# Resultados Experimentais - mppSort GPU
# Data: ...
# GPU: ...
# Parametros: h=256 bins, nR=10 repeticoes

Nro_Elementos,Vazao_mppSort_GElements_s,Vazao_Thrust_GElements_s,Speedup,Tempo_Medio_ms
1000000,X.XXX,Y.YYY,Z.ZZ,W.WWW
2000000,X.XXX,Y.YYY,Z.ZZ,W.WWW
4000000,X.XXX,Y.YYY,Z.ZZ,W.WWW
8000000,X.XXX,Y.YYY,Z.ZZ,W.WWW
```

---

## 📊 Interpretação dos Resultados

### Vazão (Throughput)
- Medida em GElements/s (bilhões de elementos por segundo)
- Quanto maior, melhor
- Indica quantos elementos são ordenados por segundo

### Speedup (Aceleração)
- `Speedup = Vazão_Thrust / Vazão_mppSort`
- **Speedup < 1**: mppSort é **mais rápido** que Thrust ✅
- **Speedup > 1**: Thrust é **mais rápido** que mppSort
- **Speedup ≈ 1**: Performance similar

### Observações Importantes
- O mppSort pode não ser mais rápido que Thrust em todos os casos
- O objetivo é demonstrar a implementação correta dos conceitos:
  - Histogramas paralelos
  - Soma de prefixos (scan)
  - Particionamento eficiente
  - Atomics em shared memory
- A comparação com Thrust serve como baseline e validação

---

## 🐛 Troubleshooting

### Erro: "illegal memory access"
```bash
# Recompilar e testar com tamanho pequeno
./compila.sh
./teste_debug.sh
```

### Erro: "nvcc not found"
```bash
# Adicionar CUDA ao PATH (se em cluster)
module load cuda
# ou
export PATH=/usr/local/cuda/bin:$PATH
```

### Testes muito lentos
```bash
# Usar teste rápido primeiro
./teste_rapido.sh

# Se OK, rodar experimentos completos
./gera_resultados_finais.sh
```

---

## ✅ Checklist para o Relatório

- [ ] Compilar código: `./compila.sh`
- [ ] Validar com teste pequeno: `./teste_debug.sh`
- [ ] Gerar resultados: `./gera_resultados_finais.sh`
- [ ] Copiar tabela para o relatório
- [ ] Incluir informações da GPU
- [ ] Descrever implementação dos 5 kernels
- [ ] Analisar resultados (speedup, vazão)
- [ ] Verificar que todos os testes passaram na verificação de corretude

---

## 📚 Referências

- **Especificação:** `especificacao_trabalho.txt`
- **Plano:** `plano_de_trabalho.md`
- **Código:** `mppSort.cu`
- **Compilação:** `compila.sh`

---

**Dica Final:** Use `gera_resultados_finais.sh` para uma experiência mais direta e resultados prontos para o relatório! 🚀
