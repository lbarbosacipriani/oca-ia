# Agent Notes

## Projeto

- Notebook principal: `AFIB_OUTROS_NORMAL_DATAAUG.ipynb`
- Objetivo: classificar ECG em `NORMAL`, `Other` e `AFIB`
- Estratégia atual: `K-Fold Cross Validation` com data augmentation apenas no treino

## Fluxo Atual

- Leitura de CSV com `explore_csv`
- Carregamento de imagens com `validate_and_load_images`
- Separação por `patient_id` para evitar vazamento entre treino e validação
- Treino com `simple_loop`
- Salvar métricas e modelos por fold

## Data Augmentation

- Existem duas políticas:
  - `transform1`: agressiva
  - `transform2`: conservadora para ECG
- A função `data_augumentation(..., aug_version=1|2)` controla a versão
- Para este projeto, a versão recomendada por padrão é `aug_version=2`

## Métricas

- O loop imprime:
  - loss
  - accuracy
  - f1
  - precision/recall gerais
  - precision/recall/f1 por classe
- As métricas por classe são:
  - `NORMAL`
  - `Other`
  - `AFIB`

## Atenção

- Validação não deve usar augmentation
- O split deve continuar por `patient_id`
- Augmentations muito agressivas podem degradar ECG e aumentar ruído
- Se a classe por classe ficar pior, revisar o balanceamento e a augmentação antes de mexer na arquitetura

## Convenções

- Preferir mudanças pequenas e rastreáveis no notebook
- Sempre verificar se o `.ipynb` continua válido após edição
- Salvar métricas por fold em diretório próprio
