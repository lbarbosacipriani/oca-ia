# Análise de Possíveis Causas de Overfitting 🎯

## 1. **Tamanho Pequeno do Dataset** ⚠️
- **N_samples = 200** amostras ao total
- Dividido em 5 folds = ~160 amostras de treino por fold
- Modelo ResNet50 tem milhões de parâmetros → memoriza facilmente dados pequenos
- **Impacto**: CRÍTICO

## 2. **Modelo Muito Grande para os Dados** 🤖
- **ResNet50D** (modelo pré-treinado) com 2048 features
- Apenas 1 camada de classificação (Linear: 2048 → 2 classes)
- Modelo pré-treinado congelado para features genéricas
- **Recomendação**: Usar modelos menores (MobileNet, EfficientNet-B0)

## 3. **Falta de Regularização** 🔓
- **Nenhuma regularização L1/L2** (weight decay)
- Otimizador: `Adam(lr=0.0001)` sem parâmetros de regularização
- **Solução**: Adicionar `weight_decay=1e-4` no otimizador

```python
# ❌ Atual (SEM regularização)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ✅ Melhorado (COM regularização)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
```

## 4. **Data Augmentation Limitada** 📸
- Apenas 3 transformações simples:
  - RandomVerticalFlip(p=0.5)
  - RandomRotation(degrees=30)
  - RandomHorizontalFlip(p=0.5)
- **Falta**: Zoom, blur, brightness, contrast, distortion
- **Impacto**: Modelo não generaliza bem para variações

```python
# Sugerido: Adicionar mais transformações
transforms = v2.Compose([
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=30),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Novo
    v2.GaussianBlur(kernel_size=3),                     # Novo
    v2.ColorJitter(brightness=0.2, contrast=0.2),      # Novo
])
```

## 5. **Early Stopping Desativado** 🛑
```python
# Linha 412-415 do código
#if early_stopping(val_loss, model, fold_index):
   # print(f'Treinamento parado no epoch {epoch+1}/{num_epochs}')
   # break
```
- Early Stopping está **COMENTADO**
- Treinamento continua por 100 épocas mesmo com divergência
- **Solução**: Ativar Early Stopping com `patience=10` a `15`

## 6. **Taxa de Aprendizado Muito Baixa** 📉
- `lr=0.0001` é muito conservador
- Pode causar treinamento incompleto
- **Sugerido**: Usar `lr=0.001` ou scheduler de learning rate

```python
# ✅ Com schedule de aprendizado
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
```

## 7. **Batch Size Muito Pequeno** 🎲
- `BATCH_SIZE = 20`
- Com 160 amostras de treino = ~8 batches por época
- Ruído alto nas atualizações de peso
- **Sugerido**: Aumentar para `32` ou `64` (se memória permitir)

## 8. **Sem Dropout ou BatchNorm adicional** 🎭
- Classifier padrão:
```python
nn.Sequential(
    nn.Flatten(),
    nn.ReLU(),
    nn.Linear(enet_out_size, 2)  # ← Nenhum dropout!
)
```
- **Solução**: Adicionar Dropout antes da camada linear
```python
nn.Sequential(
    nn.Flatten(),
    nn.Dropout(0.5),             # ← Novo
    nn.ReLU(),
    nn.Dropout(0.3),             # ← Novo
    nn.Linear(enet_out_size, 2)
)
```

## 9. **Critério de Parada Fraco (delta=0.0001)** ⚡
```python
t, l, _, outputs, labels = simple_loop(
    ..., 
    patience=5,      # Apenas 5 épocas
    delta=0.0001     # Threshold muito pequeno
)
```
- `delta=0.0001` permite continuação mesmo com ganho mínimo
- **Sugerido**: `delta=0.001` a `0.01` com `patience=10-15`

## 10. **Falta de Monitoramento de Métricas Específicas** 📊
- Código salva apenas raw predictions e labels
- Não calcula métricas durante treinamento (Accuracy, F1, AUC)
- Difícil detectar overfitting em tempo real
- **Solução**: Adicionar cálculo de métricas a cada época

```python
# Após cada época
from sklearn.metrics import accuracy_score, f1_score
train_acc = accuracy_score(true_labels, pred_labels)
val_acc = accuracy_score(val_true_labels, val_pred_labels)
print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
```

## 11. **Sem Validação Cruzada Adequada** 🔄
- K-Fold com 5 folds é bom
- Mas sem relatório consolidado de desempenho em todos os folds
- Difícil verificar se overfitting é consistente

## 12. **Modelo Pré-treinado Não Configurado** 🧠
```python
self.base_model = timm.create_model(
    'resnet50d.ra4_e3600_r224_in1k',
    pretrained=True  # ← Usando pesos ImageNet
)
self.features = nn.Sequential(*list(self.base_model.children())[:-1])
```
- Features estão congeladas (bom para poucos dados)
- MAS: Classificador é novo e não treinado
- **Risco**: Overfitting no classificador
- **Solução**: Fine-tuning gradual das últimas camadas

---

## 📋 **Resumo das Ações Recomendadas**

| # | Problema | Severidade | Solução |
|---|----------|-----------|--------|
| 1 | Dataset pequeno | 🔴 CRÍTICA | Coletar mais dados ou usar transfer learning |
| 2 | Sem regularização | 🔴 CRÍTICA | Adicionar `weight_decay` ao otimizador |
| 3 | Early stopping desativado | 🔴 CRÍTICA | Ativar e ajustar `patience` |
| 4 | Data augmentation limitada | 🟡 ALTA | Expandir transformações |
| 5 | Sem dropout | 🟡 ALTA | Adicionar dropout no classificador |
| 6 | Modelo muito grande | 🟡 ALTA | Usar modelos menores ou camadas adicionais |
| 7 | Sem monitoring de métricas | 🟠 MÉDIA | Calcular F1, Accuracy, AUC durante treino |
| 8 | Learning rate baixa | 🟠 MÉDIA | Aumentar ou usar scheduler |
| 9 | Batch size pequeno | 🟠 MÉDIA | Aumentar para 32-64 |
| 10 | Delta muito pequeno | 🟠 MÉDIA | Aumentar threshold para 0.001-0.01 |

---

## 🎬 **Implementação Prioritária**

```python
# 1. Ativar Early Stopping
if early_stopping(val_loss, model, fold_index):
    print(f'Treinamento parado em epoch {epoch+1}')
    break

# 2. Adicionar Regularização
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 3. Adicionar Dropout
classifier = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(enet_out_size, 2)
)

# 4. Expandir Data Augmentation
transforms = v2.Compose([
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=30),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.GaussianBlur(kernel_size=3),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
])

# 5. Aumentar BATCH_SIZE
BATCH_SIZE = 32  # De 20 para 32
```
