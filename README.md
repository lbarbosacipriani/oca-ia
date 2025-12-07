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
