# 🔧 DIAGNÓSTICO DE ERRO CUDA - Execute antes de treinar
import torch
import torch.cuda as cuda
from utils.cuda.cuda_eval import verify_mem
import pandas as pd
from PIL import Image
import numpy as np
import torch
from lib.ImageFIlter import treat_image_PIL
from sklearn.model_selection import KFold
from torch.utils.data import  DataLoader, TensorDataset, Dataset
from torch import nn
import timm
from tqdm.notebook import tqdm
import os
from utils.dataset.Subset import Subset
from utils.dataset.datasetBuild import explore_csv, validate_and_load_images, get_image_statistics
from utils.deepL.trainTestLoop import simple_loop
from utils.models.Resnet import ECGClassifierResnet
import numpy as np
from lib.writeOutputs import salvar_model


from lib.ImageFIlter import treat_image_PIL
from utils.dataset.datasetBuild import explore_csv, validate_and_load_images, get_image_statistics
from utils.deepL.trainTestLoop import simple_loop
path = '/home/leo/Documents/ecg_classifier/dataset/database_ptbxl/'

# Verificar GPU

verify_mem()

FILE_PATH = 'norm_outros_dataset.csv'


print("Iniciando o programa...")
print("Parametros de Execucao:")
folds = 5
epochs = 50
BATCH_SIZE = 128  # ✅ REDUZIDO de 64 para 32 (evita erro CUDA)
N_samples = 10000
flg_salvar_modelos = True

print(f"  Numero de folds para K-Fold Cross Validation: {folds}")

print(f"  Tamanho do batch para treinamento: {BATCH_SIZE}")

print(f"  Numero de epocas para treinamento: {epochs}")




## Exemplo de uso de resize de imagem 
path = '/home/leo/Documents/ecg_classifier/dataset/database_ptbxl/' 
image  = treat_image_PIL('12415_lr-0.png',1)
print(image)

## treat image pil as array and return to Image PIL and display
img_array = treat_image_PIL('12415_lr-0.png',3)
print(img_array.shape)
print(img_array.dtype)
img_pil = Image.fromarray(img_array.transpose(1, 2, 0))  # (3, 256, 256) → (256, 256, 3)
print(img_pil)





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))


import torch
from torch import nn
import numpy as np
import os
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import autocast, GradScaler







## Funções Otimizadas para Leitura de CSV e Carregamento de Imagens


print(f"Leitura e carregamento otimizado do arquivo '{FILE_PATH}'...\n")

# 1️⃣ EXPLORAR CSV PRIMEIRO
data = explore_csv(FILE_PATH, max_rows=10)

if data is None:
    print("❌ Não foi possível continuar. Verifique o arquivo CSV.")
else:
    # 2️⃣ RENOMEAR COLUNAS PARA PADRONIZAÇÃO
    print("\n🏷️  Padronizando nomes de colunas...")
    if data.shape[1] >= 2:
        data.rename(columns={data.columns[0]: 'path', data.columns[1]: 'label'}, inplace=True)
        print(f"✓ Colunas renomeadas: {list(data.columns[:2])}\n")
    
    # 3️⃣ CARREGAR IMAGENS COM VALIDAÇÃO
    img_dataset, data, error_count = validate_and_load_images(
        data=data,
        image_path_prefix=path,  # path definido anteriormente
        n_samples=N_samples,
        max_errors=10
    )
    
    # 4️⃣ ESTATÍSTICAS DAS IMAGENS
    get_image_statistics(img_dataset)
    
    # 5️⃣ RESUMO FINAL
    print("\n" + "="*80)
    print("✅ RESUMO FINAL DO CARREGAMENTO")
    print("="*80)
    print(f"Arquivo CSV: {FILE_PATH}")
    print(f"Total de amostras: {len(data)}")
    print(f"Shape de imagens: {img_dataset.shape}")
    print(f"Distribuição de labels:")
    print(data['label'].value_counts().to_string())
    print("="*80 + "\n")


## Data Augmentation Expandida - Compose com v2.Compose para numpy arrays (3, 256, 256)

from torchvision.transforms import v2

# Define as transformações usando v2 (Expandidas para melhor regularização e generalização)
transforms = v2.Compose([
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=30),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Zoom e translação
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),                    # Blur
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),        # Variações de cor
    v2.RandomPerspective(distortion_scale=0.2, p=0.3),                   # Perspectiva
])

# Aplica transformações a todo o dataset
print("Aplicando augmentação expandida com v2.Compose...")
img_dataset_aug = np.zeros_like(img_dataset, dtype=np.uint8)
for i in tqdm(range(len(img_dataset)), desc='Applying augmentation'):
    # Aplica transformações via PIL
    img_pil = Image.fromarray(img_dataset[i].transpose(1, 2, 0))  # (3, 256, 256) → (256, 256, 3)
    img_transformed = transforms(img_pil)
    img_dataset_aug[i] = np.array(img_transformed).transpose(2, 0, 1)  # Volta para (3, 256, 256)

tensor_imagem = torch.from_numpy(img_dataset_aug)
tensor_label = torch.tensor(data['label'].values)
data_tensor = TensorDataset(tensor_imagem, tensor_label)
print("Dataset de dados (imagens + labels) criado com sucesso. Tamanho do Dataset:", len(data_tensor))


## Distribuicao das amostras
## Histograma da distribuicao das amostras
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))  
data['label'].value_counts().plot(kind='bar')
plt.title('Distribuição das Amostras por Classe')
plt.xlabel('Classe')
plt.ylabel('Contagem de Amostras')
plt.show()



print("Tensor de Rotulos sendo gerado...")


tensor_label = torch.tensor(np.array(data['label'].astype("float32"))).unsqueeze(1)
print("Tensor de Rotulos gerado com sucesso. Tamanho do Tensor:", tensor_label.shape)

print(f"Configuracao do K-Fold para {folds} folds...")
kf = KFold(n_splits=folds)
kf.get_n_splits(tensor_imagem)
print(kf)
print("K-Fold configurado com sucesso.")

print("Iniciando o treinamento com K-Fold Cross Validation...")

train_loss_total = []
val_loss_total =[]
all_models =[]

train_dataset = TensorDataset(tensor_imagem, tensor_label)
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

    model= ECGClassifierResnet()
    if (flg_salvar_modelos):
        salvar_model(model, path='output/modelos', name_file=f'model_fold_{i}.pth')
    print(f'Train and valid for Fold {i}')
    # Treina com Early Stopping (patience=5 épocas, delta=0.001)
    t, l,_,outputs,labels = simple_loop(model, train_loader_img, val_loader_img, epochs, batch_size=BATCH_SIZE, fold_index=i, patience=5, delta=0.0001)
    ## Evaluate model.
    train_loss_total.append(t)
    val_loss_total.append(l)

print('''
      ###############################################
      Fim do treinamento com K-Fold Cross Validation
      ###############################################
      ''')
print("Treinamento com K-Fold Cross Validation concluído com sucesso.")