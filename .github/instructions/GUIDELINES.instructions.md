---
applyTo: '**'
---

**Objetivo:** Implementar o Trabalho 3 (CI1009) de Programação Paralela.

**Contexto e Arquivos de Referência:**
Você tem acesso aos seguintes arquivos que definem todo o escopo do trabalho:

1.  `especificacao_trabalho3.txt`: Os requisitos principais do trabalho a ser entregue.
2.  `especificacao_trabalho2.txt`: A especificação base para o algoritmo `mppSort`, que é um pré-requisito para a Parte B.
3.  `plano_trabalho3.md`: O plano de trabalho detalhado que estrutura a implementação e os testes.

**Instrução Mandatória:**
Siga **estritamente** o `plano_trabalho3.md` para executar a implementação. O plano já decompõe as especificações (Trab 2 e Trab 3) em fases e tarefas sequenciais.

**Tarefa:**
Seu objetivo é gerar o código-fonte em CUDA/C++ necessário para completar o projeto, conforme descrito no plano.

**Instruções Específicas para a Parte A (Segmented Sort Bitonic):**
Utilize o arquivo `segmented-sort-bitonic.cu` fornecido como esqueleto. Este arquivo contém a estrutura básica e locais marcados para inserção do seu código:

1.  **Esqueleto:** O kernel `blockBitonicSort` possui 4 seções marcadas com `>>> COLOCAR a parte X do SEU CODIGO AQUI!`. Preencha estas seções com a lógica do Bitonic Sort (carregamento com padding, ordenação bitônica, merge final e escrita).
2.  **Compilação e Testes:** Conforme indicado no `aa-README.txt`, você deve testar diferentes configurações de `THREADS_PER_BLOCK` (256, 512, 1024). O código deve ser compilado gerando binários distintos para cada configuração para encontrar a mais eficiente na máquina alvo (`nv00`).
3.  **Ambiente:** Os resultados finais devem ser reportados considerando a execução na máquina `nv00`.

**Entregáveis Esperados (baseado no plano):**
1.  O código-fonte completo da **Parte A** (`segmented-sort-bitonic.c` ou `.cu`).
2.  O código-fonte completo da **Parte B** (`mppSort.cu`), que utiliza os kernels do Trab 2 e os integra com o kernel otimizado da Parte A.

Comece executando a **Fase 1: Implementação (Parte A)** do `plano_trabalho3.md`.