# AGENTS.md

## Projeto

- Notebook principal: `AFIB_OUTROS_NORMAL_DATAAUG.ipynb`
- Objetivo: classificar ECG em `NORMAL`, `Other` e `AFIB`
- Estratégia atual: `K-Fold Cross Validation` com data augmentation apenas no treino

## Fluxo de Trabalho

- Ler o CSV com `explore_csv`
- Carregar imagens com `validate_and_load_images`
- Separar treino e validação por `patient_id`
- Treinar com `simple_loop`
- Salvar métricas e modelos por fold

## Data Augmentation

- `transform1` é a política agressiva
- `transform2` é a política conservadora para ECG
- `data_augumentation(..., aug_version=1|2)` seleciona a política
- Para baseline, preferir `aug_version=2`

## Métricas

- O treino imprime:
  - loss
  - accuracy
  - f1
  - precision e recall gerais
  - precision, recall e f1 por classe
- Classes:
  - `NORMAL`
  - `Other`
  - `AFIB`

## Atenção

- Não aplicar augmentation na validação
- Manter split por `patient_id` para evitar vazamento
- Evitar transformações agressivas demais em ECG
- Verificar se o `.ipynb` continua válido após qualquer edição

## Convenções

- Preferir mudanças pequenas e rastreáveis
- Salvar métricas por fold em diretório próprio
- Quando alterar métricas ou augmentação, validar também o impacto por classe
