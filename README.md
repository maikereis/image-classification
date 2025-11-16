# Fashion-MNIST: Classificação com Redes Neurais Densas vs Convolucionais

Implementação comparativa de classificação de imagens no dataset Fashion-MNIST usando duas arquiteturas distintas: rede totalmente conectada (densa) e rede neural convolucional (CNN).

## Dataset

Fashion-MNIST contém 70.000 imagens em escala de cinza (28×28 pixels) de itens de vestuário em 10 categorias:
- T-shirt/top, Calça, Pullover, Vestido, Casaco
- Sandália, Camisa, Tênis, Bolsa, Bota

Divisão: 60.000 imagens de treino, 10.000 de teste

## Arquiteturas Implementadas

### 1. Rede Totalmente Conectada (`01_fashion_mnist_ml.ipynb`)

**Arquitetura:**

```
Input(784)                             
↓
Dense(512, ReLU)                       
↓
Dense(512, ReLU)                       
↓
Dense(10, Softmax, float32 output)     
```

**Características:**
- ~670.000 parâmetros treináveis
- Destrói estrutura espacial via flatten
- Propensa a overfitting devido ao número de parâmetros

**Resultados:**
- Acurácia no teste: ~89,06%
- Tempo de treino: ~114s
- Inferência: ~256ms/amostra

### 2. Rede Neural Convolucional (`02_fashion_mnist_cnn.ipynb.py`)

**Arquitetura:**
```
Input(28×28×1)                                                    
↓
Conv2D(75, 3×3, ReLU)
↓
BatchNorm
↓
MaxPool(2×2, stride 2)
↓
Conv2D(50, 3×3, ReLU)
↓
Dropout(0.2)
↓
BatchNorm
↓
MaxPool(2×2, stride 2)
↓
Conv2D(25, 3×3, ReLU)
↓
BatchNorm
↓
MaxPool(2×2, stride 2)
↓
Flatten 
↓
Dense(512, ReLU) 
↓
Dropout(0.3) 
↓ 
Dense(10, Softmax, float32 output)
```

**Características:**
- ~250.000 parâmetros treináveis (redução de 62% em relação a arquitetura densa)
- Preserva relações espaciais
- Extração hierárquica de features (bordas -> padrões -> objetos)
- Melhor generalização através de compartilhamento de pesos

**Resultados:**
- Acurácia no teste: ~91%+
- Tempo de treino: ~140s
- Inferência: ~50ms/amostra

## Diferenças Arquiteturais Principais

| Aspecto | Rede Densa | CNN |
|---------|------------|-----|
| Formato de entrada | (784,) | (28, 28, 1) |
| Informação espacial | Perdida via flatten | Preservada |
| Parâmetros | Alto (~670k) | Menor (~250k) |
| Risco de overfitting | Alto | Baixo (dropout, pooling) |
| Invariância à translação | Não | Sim |
| Aprendizado de features | Global | Local -> hierárquico |

## Detalhes de Implementação Técnica

**Pré-processamento:**
- Normalização: valores de pixels de [0, 255] para [0, 1]
- Densa: achatamento para vetores 1D
- CNN: reshape para (28, 28, 1) mantendo estrutura espacial

**Configuração de Treinamento:**
- Loss: categorical crossentropy
- Otimizador: Adam
- Batch size: 32
- Épocas: 20
- Mixed precision (FP16) habilitada para aceleração em GPU

**Técnicas de Regularização (CNN):**
- Batch Normalization: reduz internal covariate shift
- Dropout: taxas de 0.2 e 0.3 em diferentes camadas
- MaxPooling: redução espacial para robustez

## Requisitos

```
uv sync
```