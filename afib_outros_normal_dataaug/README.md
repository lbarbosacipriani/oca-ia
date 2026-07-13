# AFIB Outros Normal DataAug

Este diretório contém versões do pipeline para classificação de ECG em `NORMAL`, `Other` e `AFIB`, com variações de augmentation, regularização e estratégia de treino.

## Visão Geral

- O objetivo comum entre os notebooks é treinar e avaliar um classificador multi-classe com `K-Fold Cross Validation`.
- O split é feito por `patient_id` para evitar vazamento entre treino e validação.
- A augmentation deve ser aplicada apenas no treino.
- Os resultados de cada execução são salvos em `output/`, incluindo métricas, logs e documentação da execução.

## Notebooks

### `AFIB_OUTROS_NORMAL_DATAAUG.ipynb`

Versão base do projeto.

- Mantém o pipeline original de treino e validação.
- Inclui salvamento de métricas por fold.
- Gera `readme.md` e `training_log.txt` em `output/`.
- Serve como referência para comparar os demais experimentos.

### `AFIB_OUTROS_NORMAL_DATAAUG_regularizado.ipynb`

Versão com regularização mais forte e augmentation ajustada para reduzir overfitting.

- Adiciona dropout leve no classificador.
- Usa augmentation mais variada, mas ainda conservadora para ECG.
- Mantém validação sem `shuffle`.
- Registra a evolução das métricas em `output/training_log.txt`.
- Salva documentação da execução em `output/readme.md`.

### `AFIB_OUTROS_NORMAL_DATAAUG_freeze_backbone.ipynb`

Versão mais conservadora, com congelamento parcial do backbone no início do treino.

- Congela o backbone nas primeiras épocas.
- Destrava apenas parte final da rede depois de um número configurável de épocas.
- Mantém a cabeça com dropout leve.
- Indicado quando o gap entre treino e validação ainda é alto.

### `AFIB_OUTROS_NORMAL_DATAAUG_refatorado.ipynb`

Versão refatorada do pipeline.

- Organiza o fluxo de treino de forma mais modular.
- Usa `GroupKFold`/`Group split` por `patient_id` em vez de split ingênuo.
- Separa melhor dataset, treino e validação.
- Serve como base para futuras manutenções e experimentos mais limpos.

## Estrutura de Saída

Cada execução grava artefatos em uma pasta de saída do tipo:

```text
output/
├── readme.md
├── training_log.txt
└── metricas/
    ├── train/
    ├── valid/
    └── class_metrics/
```

Conteúdo importante:

- `readme.md`: resumo da execução e parâmetros usados.
- `training_log.txt`: evolução época a época com métricas de treino, validação e por classe.
- `metricas/train`: predições e rótulos do treino.
- `metricas/valid`: predições e rótulos da validação.
- `metricas/class_metrics`: métricas por classe e gráficos correspondentes.

## Recomendações

- Para baseline, usar `AFIB_OUTROS_NORMAL_DATAAUG_regularizado.ipynb`.
- Se o overfitting persistir, testar `AFIB_OUTROS_NORMAL_DATAAUG_freeze_backbone.ipynb`.
- Evitar augmentation agressiva demais em ECG.
- Manter a validação sem augmentation e com split por `patient_id`.

