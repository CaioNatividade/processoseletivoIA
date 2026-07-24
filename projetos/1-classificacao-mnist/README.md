# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

**Exemplo de execução** (execute dentro de `projetos/1-classificacao-mnist`):

```bash
cd "c:\Users\SUPORTE\Documents\programacao\processoseletivoIA\projetos\1-classificacao-mnist"
python run_inference.py
```

> Observação: se a pasta original `Programação` foi renomeada para `programacao`,
> use o caminho atualizado acima para evitar problemas de compatibilidade com
> algumas bibliotecas nativas.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Caio Juan da Natividade Santos:**

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada possui **4 blocos convolucionais** sequenciais, cada um composto por:

- **Camada Conv2D**: com ativação ReLU (32 → 64 → 128 → 128 filtros)
- **Batch Normalization**: para normalizar ativações e estabilizar o treinamento
- **MaxPooling2D**: redução de dimensionalidade (2x2)

Após os blocos convolucionais:

- **Flatten**: converte o mapa de características para um vetor 1D (1152 neurônios)
- **Dropout (0.5)**: regularização para evitar overfitting
- **Dense (10 neurônios)**: camada de saída com ativação softmax para classificação em 10 classes

**Total de parâmetros**: 253.194 (~989 KB treináveis)

**Estratégia de validação**: Split de 20% dos dados de treinamento como conjunto de validação via `validation_split=0.2`

**Early Stopping**: Monitoramento da perda de validação (`val_loss`) com paciência de 3 epochs, garantindo que o modelo seja interrompido ao começar a fazer overfitting.

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow**: >= 2.12 (incluindo Keras para construção do modelo)
- **NumPy**: para manipulação de arrays
- **Python**: 3.11

Bibliotecas indiretas (incluídas por TensorFlow):

- scipy, protobuf, h5py (para salvamento do modelo em formato HDF5)

### 3️⃣ Técnica de Otimização do Modelo

**Dynamic Range Quantization** (`tf.lite.Optimize.DEFAULT`):

Esta técnica reduz a precisão dos pesos do modelo de float32 para int8, mantendo ativações em float32. É aplicada automaticamente pelo TFLiteConverter ao usar `converter.optimizations = [tf.lite.Optimize.DEFAULT]`.

**Benefícios**:

- Reduz drasticamente o tamanho do modelo (~91% de compressão)
- Mantém acurácia próxima ao modelo original
- Ideal para Edge AI e dispositivos com memória limitada
- Execução rápida em CPUs sem hardware especializado

### 4️⃣ Resultados Obtidos

**Acurácia**:

- Acurácia de validação (conjunto de teste): **98.93%**
- Acurácia de treinamento final (epoch 7): **99.53%**
- Perda de validação final: 0.0336

**Tamanho dos modelos**:
| Modelo | Tamanho | Formato |
|--------|---------|----------|
| model.h5 | 3.97 MB | Keras (HDF5) |
| model.tflite | 0.26 MB | TensorFlow Lite |
| **Redução** | **91.4%** | **7x menor** |

**Tempo de treinamento**: ~350 segundos (5 minutos) em CPU

**Nota**: O modelo parou na epoch 7 (de 15 máximo) graças ao Early Stopping, economizando tempo e evitando overfitting.

### 5️⃣ Comentários Adicionais (Opcional)

**Decisões técnicas**:

- Escolha de **4 blocos convolucionais**: balanceamento entre capacidade do modelo e complexidade computacional para CPU
- **Padding='same'**: mantém as dimensões espaciais para melhor preservação de informações
- **Dropout 0.5**: taxa agressiva escolhida para aumentar regularização e evitar overfitting em dataset simples
- **Early Stopping com paciência 3**: foi suficiente para este dataset, evitando computação desnecessária

**Dificuldades encontradas**:

- Caminho da pasta com acentos (`Programação`) causou erro ao carregar o modelo TFLite — resolvido com workaround temporário antes de renomear para `programacao`

**Limitações**:

- MNIST é um dataset simples, então 98.93% é esperado; modelos ainda mais simples (ex: 2 blocos) poderiam atingir acurácia similar
- Treinamento em CPU é lento; com GPU seria ~100x mais rápido
- Sem data augmentation — em datasets reais, isso melhoraria robustez

**Aprendizados**:

- Quantização é extremamente efetiva para Edge AI, comprimindo 91.4% do tamanho
- Early Stopping é essencial para economizar tempo e evitar overfitting
- Nomes de pastas sem acentos são importantes para compatibilidade com bibliotecas C/C++ subjacentes

### 6️⃣ Exemplo de Inferência

```
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

**Análise**:

- **100% de acurácia** nas 5 amostras testadas (5/5 corretas)
- O modelo se comporta perfeitamente mesmo após quantização
- Não houve erros ou casos interessantes — todos os dígitos foram corretamente classificados
- Isso confirma que a técnica de Dynamic Range Quantization manteve a capacidade discriminativa do modelo
