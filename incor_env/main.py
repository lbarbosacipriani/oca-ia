import os


FILE_PATH = 'dataset/oca_incor.csv'


print("Iniciando o programa...")
import pandas as pd
import numpy as np
import torch
from lib.ImageFIlter import treat_image_PIL
from sklearn.model_selection import KFold
from torch.utils.data import  DataLoader, TensorDataset
import modelos.ECGClassifierResnet as   resnet_model
from lib.subset import Subset
from simple_loop import simple_loop, salvar_model

print(f"Leitura dos dados no arquivo '{FILE_PATH}'...")

data = pd.read_csv(FILE_PATH)
data.rename(columns={data.columns[0]:'path'}, inplace=True)
data.rename(columns={data.columns[1]:'label'}, inplace=True)
print("Dados lidos com sucesso. Tamnaho dos dados:", data.shape)

print("Geracao Tensor de Imagens...")
## Loop
img_dataset = np.ones((data.shape[0],3,256,256),dtype=np.uint8)

j=0
for i in data['path']:
    img_dataset[j]=treat_image_PIL('dataset/path/'+i,2)
    j+=1
tensor_imagem = torch.tensor(img_dataset)
print("Tensor de Imagens gerado com sucesso. Tamanho do Tensor:", tensor_imagem.shape)



print("Tensor de Rotulos sendo gerado...")


tensor_label = torch.tensor(np.array(data['label'].astype(int)))
print("Tensor de Rotulos gerado com sucesso. Tamanho do Tensor:", tensor_label.shape)

folds = 10
print(f"Configuracao do K-Fold para {folds} folds...")
kf = KFold(n_splits=folds)
kf.get_n_splits(tensor_imagem)
print(kf)
print("K-Fold configurado com sucesso.")

print("Iniciando o treinamento com K-Fold Cross Validation...")

train_loss_total = []
val_loss_total =[]
all_models =[]
epochs = 10
BATCH_SIZE = 5
train_dataset = TensorDataset(tensor_imagem, tensor_label)
## create first model.
print('''
      ###############################################
      Inicio do treinamento com K-Fold Cross Validation
      ###############################################
      ''')
for i, (train_index, test_index) in enumerate(kf.split(train_dataset)):
    print(f"Fold {i}:")
    #print(f"  Train: index={train_index}")
    #print(f"  Test:  index={test_index}")
    ## init train test for folder
    train_dataset_part = Subset( train_dataset, train_index)
    val_dataset_part = Subset( train_dataset, test_index)

    train_loader_img = DataLoader(train_dataset_part, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_img = DataLoader(val_dataset_part, batch_size=BATCH_SIZE, shuffle=True)

    model= resnet_model.ECGClassifierResnet( num_classes=1)
    salvar_model(model, path='output/modelos', name_file=f'model_fold_{i}.pth')
    print(f'Train and valid for Fold {i}')
    t, l,_,outputs,labels = simple_loop(model, train_loader_img,val_loader_img, epochs, batch_size = BATCH_SIZE, fold_index =i)
    ## Evaluate model.
    train_loss_total.append(t)
    val_loss_total.append(l)

print('''
      ###############################################
      Fim do treinamento com K-Fold Cross Validation
      ###############################################
      ''')
print("Treinamento com K-Fold Cross Validation concluído com sucesso.")