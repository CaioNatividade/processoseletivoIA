# Projeto 1 — Classificação MNIST

## 📝 Relatório do Candidato

👤 **Caio Juan da Natividade Santos:**

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada possui **4 blocos convolucionais** sequenciais:

- **Blocos 1 a 3**: `Conv2D` com ativação ReLU (32 → 64 → 128 filtros),
  `BatchNormalization` e `MaxPooling2D` (2x2)
- **Bloco 4**: `Conv2D` com 128 filtros e ativação ReLU, seguido de
  `BatchNormalization`

Após os blocos convolucionais:

- **Flatten**: converte o mapa de características para um vetor 1D (1152 neurônios)
- **Dropout (0.5)**: regularização para evitar overfitting
- **Dense (10 neurônios)**: camada de saída com ativação softmax para classificação em 10 classes

**Total de parâmetros**: 253.194

**Estratégia de validação**: Split de 20% dos dados de treinamento como conjunto de validação via `validation_split=0.2`

**Early Stopping**: Monitoramento da perda de validação (`val_loss`) com paciência de 3 epochs, garantindo que o modelo seja interrompido ao começar a fazer overfitting.

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow**: 2.12 (incluindo Keras para construção do modelo)
- **NumPy**: para manipulação de arrays
- **Python**: 3.11

Bibliotecas indiretas (incluídas por TensorFlow):

- scipy, protobuf, h5py (para salvamento do modelo em formato HDF5)

### 3️⃣ Técnica de Otimização do Modelo

**Dynamic Range Quantization** (`tf.lite.Optimize.DEFAULT`):

Esta técnica aplica quantização dinâmica aos pesos durante a conversão para TensorFlow Lite, reduzindo significativamente o tamanho do modelo e melhorando a eficiência para execução em dispositivos com recursos limitados, mantendo boa precisão na maioria dos casos.

**Benefícios**:

- Reduz drasticamente o tamanho do modelo (~91% de compressão)
- Mantém acurácia próxima ao modelo original
- Ideal para Edge AI e dispositivos com memória limitada
- Execução rápida em CPUs sem hardware especializado

### 4️⃣ Resultados Obtidos

**Acurácia**:

- Acurácia do modelo Keras no conjunto de teste: **99,09%**
- Perda do modelo Keras no conjunto de teste: **0,0304**
- Acurácia do modelo TFLite nas 10.000 amostras de teste: **99,09%**

**Tamanho dos modelos**:
| Modelo | Tamanho | Formato |
|--------|---------|----------|
| model.h5 | 2,97 MiB (3.117.336 bytes) | Keras (HDF5) |
| model.tflite | 0,25 MiB (267.176 bytes) | TensorFlow Lite |
| **Redução** | **91,43%** | **11,67x menor** |

Os resultados acima foram medidos diretamente nos artefatos `model.h5` e
`model.tflite`. O treinamento foi configurado para no máximo 15 épocas, com
interrupção antecipada após 3 épocas sem melhora da perda de validação.

### 5️⃣ Comentários Adicionais (Opcional)

**Decisões técnicas**:

- Escolha de **4 blocos convolucionais**: balanceamento entre capacidade do modelo e complexidade computacional para CPU
- **Padding='same'**: mantém as dimensões espaciais para melhor preservação de informações
- **Dropout 0.5**: taxa agressiva escolhida para aumentar regularização e evitar overfitting em dataset simples
- **Early Stopping com paciência 3**: foi suficiente para este dataset, evitando computação desnecessária

**Dificuldades encontradas**:

- Caminho da pasta com acentos (`Programação`) causou erro ao carregar o modelo TFLite — resolvido com workaround temporário antes de renomear para `programacao`

**Limitações**:

- MNIST é um dataset simples; modelos ainda mais compactos poderiam atingir acurácia similar
- O treinamento em CPU é mais demorado do que em hardware acelerado
- Sem data augmentation — em datasets reais, isso melhoraria robustez

**Aprendizados**:

- Quantização é efetiva para Edge AI, reduzindo o tamanho do artefato em 91,43%
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
- O modelo classificou corretamente as cinco amostras exibidas após a quantização
- Não houve erros ou casos interessantes — todos os dígitos foram corretamente classificados
- No conjunto de teste completo, o modelo TFLite obteve 99,09% de acurácia,
  indicando que a quantização preservou o desempenho observado no modelo Keras
