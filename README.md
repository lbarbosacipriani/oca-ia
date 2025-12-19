# oca-ia

Repositorio com modelos de Deep Learning para classificacao de Oclusao Coronaria Aguda. 
dataset:

Image_id: identificador único da imagem,
ID: identificador único do paciente (repare que um paciente pode ter múltiplos exames de ECG)
IDADE: idade do paciente em anos no momento da coleta do exame,
SEXO: sexo do paciente (M: masculino, F: feminino, D ou O: outro)
ETNIA: etnia do paciente (ALBINO, AMARELA, BRANCA, NEGRA, PARDA, UNKNOWN, VERMELHO) - UNKNOWN: desconhecido

Depois disso, temos uma coluna para cada classe do banco de dados. Nesse banco de dados em específico:
FA,TA/Flutter,TPSV/TS,TV/FV,BAV 2º/3º/Avanc./BS,Supra ST,Corrente de Lesao,Extrassistole,BRD,BRE,MP,Normal,Outros,Exclusão

Se a coluna tiver o valor 'True' aquele exame apresenta aquela classe, caso contrário, aquele exame não apresenta aquela classe. Repare que um mesmo exame pode ter diferentes classes positivas (problema multilabel)


Modelos utiliZados para benchmark

elfficient net
resnet 50
mamba


## Historico de altercoes

### 07-12-25

versao estavel com metricas inseridas para calculo posterior no path /output


### To do list

- Implementar tratamento de vetores predict_label para ter tamanho variavel de acordo com batch size. 



# Documentacao 

### TRAIN–VALID LOOP


Inputs:
- modelo
- dataloader_train
- dataloader_test
- epocas

-------------------------------------
### PARA CADA ÉPOCA
-------------------------------------

### TREINAMENTO
-----------
Para cada batch em dataloader_train:

    1) Ler imagens e rótulos

    2) Forward → outputs

    3) Calcular loss

    4) Backward (loss.backward)

    5) optimizer.step()

### VALIDAÇÃO
---------
Para cada batch em dataloader_test:

    1) Ler imagens e rótulos
    2) Forward → outputs
    3) Calcular loss

### FINAL DA ÉPOCA
--------------
- Calcular loss médio de treino
- Calcular loss médio de validação
- Exibir métricas da época

(Repete até terminar as épocas)


### Estrutura arquivos de output

train > fold _ {Numero do fold} > npy arquivo com ( epocas, total_base/batch (quantidade de loops de batch), batch_size).  