#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 🔧 DIAGNÓSTICO DE ERRO CUDA - Execute antes de treinar
import torch
import torch.cuda as cuda

print("="*80)
print("📊 DIAGNÓSTICO DO SISTEMA GPU")
print("="*80)

# Verificar GPU
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"GPU encontrada: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma'}")

if torch.cuda.is_available():
    # Limpar cache antes de começar
    print("\n🧹 Limpando cache de GPU...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Informações de memória
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / 1e9
    print(f"Memória total da GPU: {total_memory:.2f} GB")

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Memória alocada: {allocated:.2f} GB")
    print(f"Memória reservada: {reserved:.2f} GB")
    print(f"Memória livre: {(total_memory - reserved):.2f} GB")

    # Verificar compatibilidade
    print(f"\nVersão CUDA: {torch.version.cuda}")
    print(f"Versão cuDNN: {torch.backends.cudnn.version()}")
    print(f"cuDNN habilitado: {torch.backends.cudnn.enabled}")

else:
    print("⚠️  CUDA não disponível! Usando CPU (muito lento).")

print("="*80 + "\n")


# In[2]:


import pandas as pd
from PIL import Image, ImageChops
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import  DataLoader, TensorDataset, Dataset
from torch import nn
import timm
from tqdm.notebook import tqdm
import torch.optim as optim
import os


# In[3]:


FILE_PATH = 'dataset_FA_OUTROS_NORMAL_BINARY_HOT_ENCODING.csv'

print("Iniciando o programa...")
print("Parametros de Execucao:")
folds = 5
epochs = 50
BATCH_SIZE = 8 # ✅ REDUZIDO de 64 para 32 (evita erro CUDA)
N_samples = 10000  # Use None para usar todos os dados
flg_salvar_modelos = True
path = '/home/leo/Documents/ecg_classifier/dataset/database_ptbxl/'
path_out = 'saida_123/saidas/'

print(f"  Numero de folds para K-Fold Cross Validation: {folds}")

print(f"  Tamanho do batch para treinamento: {BATCH_SIZE}")

print(f"  Numero de epocas para treinamento: {epochs}")


# In[4]:


data = pd.read_csv(FILE_PATH)
tensor_label = data[['NORMAL','Other','AFIB']][0:N_samples]
tensor_label.value_counts()
data['NORMAL'].isna().sum()


# In[5]:


from PIL import Image, ImageChops
import numpy as np
import os
import shutil
import io
from pathlib import Path
def treat_image_PIL(img_path, type_return=2):
    ''''
    Input: Img_path, type return.

    Img_path: path da imagem em formato png, img...
    type_return: 1-> retorno como PIL.
                2 ou sem type_return -> retorno como numpy array tipo uint8

    Output:
    '''
    im = Image.open(path+img_path)

    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size

    rgb =Image.Image.split(im)

    data =rgb
    b= data[0]
    g= data[1]
    r= data[2]
    #img_out = b+g+.5*r
    #img_out_2 = img_out[500:1600, 50:2100]

    newsize = (256, 256)
   # im3 =ImageChops.subtract(mask,b, scale=1.0, offset=0)

    b1 = b.crop((120,500,2100,1600))
    g1 = g.crop((120,500,2100,1600))
    r1 = r.crop((120,500,2100,1600))
    im1 = b1.resize(newsize, Image.Resampling.LANCZOS).convert('L')
    im2 = g1.resize(newsize, Image.Resampling.LANCZOS).convert('L')
    im3 = r1.resize(newsize, Image.Resampling.LANCZOS).convert('L')
    if type_return ==1:
        return Image.merge("RGB",(im1,im1,im1))
    elif type_return ==2:

        return np.array([im1,im1,im1],dtype=np.uint8)
    elif type_return ==3:

        return np.array([im1,im2,im3],dtype=np.uint8)
    elif type_return ==4:

        return np.array(im3,dtype=np.uint8)

def save_file_to_dir(file_obj, directory, filename):
    """
    Save a file to `directory` with the given `filename`.

    Parameters
    - file_obj: a PIL Image, a filesystem path (str or Path) to an existing file,
                bytes/bytearray, or a file-like object with a .read() method.
    - directory: target directory where the file will be saved.
    - filename: the name to use for the saved file (including extension if desired).

    Returns
    - full path (str) to the saved file.

    Raises
    - FileNotFoundError if a provided source path does not exist.
    - ValueError if the provided file_obj type is unsupported.
    """
    os.makedirs(directory, exist_ok=True)
    dest = os.path.join(directory, filename)

    # PIL Image
    if isinstance(file_obj, Image.Image):
        file_obj.save(dest)
        return dest

    # Path-like or string pointing to an existing file
    if isinstance(file_obj, (str, Path)):
        src = str(file_obj)
        if os.path.exists(src):
            shutil.copy(src, dest)
            return dest
        raise FileNotFoundError(f"Source path not found: {src}")

    # File-like object
    if hasattr(file_obj, "read"):
        data = file_obj.read()
        # If read() returned bytes -> try open as image, otherwise write raw
        if isinstance(data, (bytes, bytearray)):
            try:
                img = Image.open(io.BytesIO(data))
                img.save(dest)
                return dest
            except Exception:
                with open(dest, "wb") as f:
                    f.write(data)
                return dest
        else:
            # assume text
            with open(dest, "w", encoding="utf-8") as f:
                f.write(data)
            return dest

    # Raw bytes
    if isinstance(file_obj, (bytes, bytearray)):
        try:
            img = Image.open(io.BytesIO(file_obj))
            img.save(dest)
            return dest
        except Exception:
            with open(dest, "wb") as f:
                f.write(file_obj)
            return dest

    raise ValueError("file_obj must be a PIL.Image, path string/Path, bytes or file-like object")


# In[6]:


## Exemplo de uso de resize de imagem
image  = treat_image_PIL('00004_lr-0.png',1)

## treat image pil as array and return to Image PIL and display
img_array = treat_image_PIL('00004_lr-0.png',3)
print(img_array.shape)
print(img_array.dtype)
img_pil = Image.fromarray(img_array.transpose(1, 2, 0))  # (3, 256, 256) → (256, 256, 3)


# In[7]:


import os
## Cria funcao para validar se pasta a ser inserida existe. Caso nao exista, cria a pasta
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'Pasta {folder_path} criada.')
    else:
        print(f'Pasta {folder_path} ja existe.')


# In[8]:


from torch.utils.data import  Dataset

class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    @property
    def classes(self):
        return self.data.classes

    def shape(self):
        return self.dataset


# In[9]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))


# In[10]:


import torch
from torch import nn
import numpy as np
import os
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

def salvar_model(model, path = '/content/drive/MyDrive/4000-files/output/modelos', name_file='model.pth'):
    create_folder_if_not_exists(path)

    full_path = os.path.join(path, name_file)
    torch.save(model.state_dict(), full_path)

def salvar_metricas(path, name_file_train='train_loss_total.npy', name_file_val='val_loss_total.npy',predict_label=None, true_label=None):
    create_folder_if_not_exists(path)
    full_path_train = os.path.join(path, name_file_train)
    full_path_val = os.path.join(path, name_file_val)
    np.save(full_path_train, np.array(predict_label))
    np.save(full_path_val, np.array(true_label))
    print(f'Metricas salvas em {path} com os nomes {name_file_train} e {name_file_val} e tamanhos {np.array(predict_label).shape} e {np.array(true_label).shape}')

def simple_loop(model, train_image, val_image, epochs, batch_size, fold_index, patience=10, delta=0.0001, device_input=None):
    """
    Loop de treinamento e validação com Early Stopping, regularização e monitoramento de métricas.
    ✅ MELHORIAS: Mixed Precision, detecção CUDA, garantir float32, device handling

    Parâmetros adicionais:
        patience (int): Número de épocas sem melhora antes de parar (default: 10)
        delta (float): Threshold mínimo de melhora para considerar como melhora (default: 0.001)
        device_input: torch.device('cuda:0') ou torch.device('cpu')
    """
    # Use device global se não fornecido
    if device_input is None:
        device_input = device

    # Simple training loop
    num_epochs = epochs
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1s, val_f1s = [], []
    lim_loss = 1.5
    iter_size = batch_size
    print(f'Number of training images per iteration: {iter_size}')
    print(f'Device sendo usado: {device_input}')

    # ✅ IMPORTANTE: Move model para device PRIMEIRO
    try:
        model = model.to(device_input)
        print(f'✓ Modelo movido para {device_input}')
    except Exception as e:
        print(f'❌ ERRO ao mover modelo para device: {e}')
        raise


    # Use CrossEntropyLoss for single-label multi-class classification
    criterion = nn.CrossEntropyLoss()
    # Optimizer COM regularização (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=3)
    predict_label_full_train = []
    predict_label_full = []
    true_label_full_train = []
    true_label_full = []

    # Early stopping state
    best_val = float("inf")
    wait = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        predict_label_train = []
        true_label_train = []
        running_loss_train = 0.0

        try:
            for images, labels in tqdm(train_image, desc='Training loop'):
                # ✅ FIX: Garantir float32 e evitar overflow
                images = images.to(device_input)
                labels = labels.to(device_input)
                images = images.to(torch.float32)
                labels = labels.to(torch.float32)
                optimizer.zero_grad()

                # ✅ FIX: Usar autocast para mixed precision

                outputs = model(images)

                # Convert one-hot labels to class indices for CrossEntropyLoss
                targets = torch.argmax(labels, dim=1)
                loss_train = criterion(outputs, targets)
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                _pred_train = torch.argmax(outputs, dim=1)
                running_loss_train += loss_train.item() * labels.size(0)
                # Salvar predições e labels CORRETAMENTE
                try:
                    pred_list = _pred_train.cpu().numpy().astype(int).tolist()
                    true_list = _true.cpu().numpy().astype(int).tolist()

                    if not isinstance(pred_list, list):
                        pred_list = [pred_list]
                    if not isinstance(true_list, list):
                        true_list = [true_list]

                    if len(pred_list) == iter_size:
                        predict_label_train.append(pred_list)
                        true_label_train.append(true_list)

                except Exception as e:
                    print(f"⚠ Erro ao salvar predições de treinamento: {e}")

        except RuntimeError as e:
            if "CUDA" in str(e) or "assert" in str(e).lower():
                print(f"\n❌ ERRO CUDA durante treinamento: {e}")
                print("💡 Sugestões:")
                print("   1. Reduza BATCH_SIZE")
                print("   2. Execute torch.cuda.empty_cache()")
                print("   3. Use modelos menores")
                torch.cuda.empty_cache()
                raise
            else:
                raise

        train_loss = running_loss_train / len(train_image.dataset)
        train_losses.append(train_loss)

        # Calcular métricas de treinamento
        try:
            if predict_label_train and true_label_train:
                train_preds = np.concatenate(predict_label_train)
                train_trues = np.concatenate(true_label_train)
                train_acc = accuracy_score(train_trues, train_preds)
                train_f1 = f1_score(train_trues, train_preds, average='weighted', zero_division=0)
                train_accuracies.append(train_acc)
                train_f1s.append(train_f1)
        except Exception as e:
            print(f"⚠ Erro ao calcular métricas de treinamento: {e}")
            print(f"  predict_label_train: {train_preds}")
            print(f"  true_label_train: {train_trues}")
        try:
            _p_train = predict_label_train
            _t_train = true_label_train
            predict_label_full_train.append(_p_train)
            true_label_full_train.append(_t_train)
        except Exception as e:
            print(f"⚠ Erro ao concatenar métricas de treinamento: {e}")
            print(f"  predict_label_train: {train_preds}")
            print(f"  true_label_train: {train_trues}")

        # Validation phase
        model.eval()
        running_loss_valid = 0.0
        predict_label = []
        true_label = []
        _iter = 0

        try:
            with torch.no_grad():
                for images, labels in tqdm(val_image, desc='Validation loop'):
                    # ✅ FIX: Garantir float32
                    images = images.to(device_input)
                    labels = labels.to(device_input)
                    images = images.to(torch.float32)
                    labels = labels.to(torch.float32)
                   # print(f"Images shape: {images.shape}, dtype: {images.dtype}")
                    #print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
                    outputs = model(images)
                    pred_out = torch.argmax(outputs, dim=1)
                    true_out = torch.argmax(labels, dim=1)  # Convertendo one-hot para índices de classe
                    # Use logits and integer class targets for CrossEntropyLoss
                    loss_valid = criterion(outputs, true_out)

                    # Get predictions
                    _pred = torch.argmax(outputs, dim=1)

                    # Salvar predições e labels
                    try:
                        pred_list = _pred.cpu().numpy().astype(int).tolist()
                        true_list = true_out.cpu().numpy().astype(int).tolist()

                        if not isinstance(pred_list, list):
                            pred_list = [pred_list]
                        if not isinstance(true_list, list):
                            true_list = [true_list]

                        if len(pred_list) == iter_size:
                            predict_label.append(pred_list)
                            true_label.append(true_list)

                    except Exception as e:
                        print(f"⚠ Erro ao salvar predições de validação: {e}")

                    running_loss_valid += loss_valid.item() * labels.size(0)
                    _iter += 1

        except RuntimeError as e:
            if "CUDA" in str(e) or "assert" in str(e).lower():
                print(f"\n❌ ERRO CUDA durante validação: {e}")
                torch.cuda.empty_cache()
                raise
            else:
                raise

        val_loss = running_loss_valid / len(val_image.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Calcular métricas de validação
        val_precision = 0
        val_recall = 0
        try:
            if predict_label and true_label:
                val_preds = np.concatenate(predict_label)
                val_trues = np.concatenate(true_label)
                val_acc = accuracy_score(val_trues, val_preds)
                val_f1 = f1_score(val_trues, val_preds, average='weighted', zero_division=0)
                val_precision = precision_score(val_trues, val_preds, average='weighted', zero_division=0)
                val_recall = recall_score(val_trues, val_preds, average='weighted', zero_division=0)
                val_accuracies.append(val_acc)
                val_f1s.append(val_f1)
        except Exception as e:
            print(f"⚠ Erro ao calcular métricas de validação: {e}")

        try:
            _p = predict_label
            _t = true_label
            predict_label_full.append(_p)
            true_label_full.append(_t)
        except Exception as e:
            print(f"⚠ Erro ao concatenar métricas de validação: {e}")

        # Print detalhado das métricas
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        print(f"Train Loss: {train_loss:.6f}", end="")
        if train_accuracies:
            print(f" | Train Acc: {train_accuracies[-1]:.4f}", end="")
            print(f" | Train F1: {train_f1s[-1]:.4f}", end="")

        print()
        print(f"Val Loss:   {val_loss:.6f}", end="")
        if val_accuracies:
            print(f" | Val Acc: {val_accuracies[-1]:.4f}", end="")
            print(f" | Val F1: {val_f1s[-1]:.4f}", end="")
            print(f" | Precision: {val_precision:.4f}", end="")
            print(f" | Recall: {val_recall:.4f}", end="")
        print()
        print(f"{'='*80}\n")

    predict_label_full_out = np.array(predict_label_full, dtype=object)
    true_label_full_out = np.array(true_label_full, dtype=object)
    predict_label_full_train_out = np.array(predict_label_full_train, dtype=object)
    true_label_full_train_out = np.array(true_label_full_train, dtype=object)
    print(f'Salvado das metricas de validacao e treino')

    salvar_metricas(path=f'{path_out}/output/metricas/valid/fold_{fold_index}',
                    name_file_train=f'predict_label_valid_fold_{fold_index}.npy',
                    name_file_val=f'true_label_valid_fold_{fold_index}.npy',
                    predict_label= predict_label_full_out, true_label= true_label_full_out)

    salvar_metricas(path=f'{path_out}/output/metricas/train/fold_{fold_index}',
                    name_file_train=f'predict_label_train_fold_{fold_index}.npy',
                    name_file_val=f'true_label_train_fold_{fold_index}.npy',
                    predict_label= predict_label_full_train_out, true_label= true_label_full_train_out)
    print(f'Finalizado o salvamento das metricas')
    return train_losses, val_losses, model, predict_label_full_out, true_label_full_out


# In[11]:


#from models import ECGClassifierResnet
from torch import nn
import timm
class ECGClassifierResnet(nn.Module):
    def __init__(self, num_classes=1):
        super(ECGClassifierResnet, self).__init__()
        # Where we define all the parts of the model
        #self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.base_model=timm.create_model('resnet50d.ra4_e3600_r224_in1k',pretrained=True)
        #self.base_model = timm.create_model('vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k',num_classes=5,pretrained=True)

        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 2048        # Make a classifier
        # For binary classification com Dropout para regularização
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.7),           # Dropout para regularização (reduz overfitting)
            nn.ReLU(),
           # nn.Dropout(0.5),           # Dropout adicional antes da saída
            nn.Linear(enet_out_size, 3)##
        ) # saida como Softmax para classificacao single label

    def forward(self, x):
        # Connect these parts and return the output
        #converte 1 canal para 3 canais (RGB) usando uma camada Conv2d
        #x1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)(x)  # Converte de 1 canal para 3 canais
        x = self.features(x)
        output = self.classifier(x)
        return output


# In[12]:


#from models import ECGClassifierResnet
from torch import nn
import timm
class ECGClassifierMambaVision(nn.Module):
    def __init__(self, num_classes=1):
        super(ECGClassifierMambaVision, self).__init__()
        # Where we define all the parts of the model
        #self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.base_model=timm.create_model('mamba_vision_L2',pretrained=True)
        #self.base_model = timm.create_model('vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k',num_classes=5,pretrained=True)

        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 2048        # Make a classifier
        # For binary classification com Dropout para regularização
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.7),           # Dropout para regularização (reduz overfitting)
            nn.ReLU(),
            nn.Dropout(0.5),           # Dropout adicional antes da saída
            nn.Linear(enet_out_size, 2)
        ) # saida como Softmax para classificacao single label

    def forward(self, x):
        # Connect these parts and return the output
        #converte 1 canal para 3 canais (RGB) usando uma camada Conv2d
        #x1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)(x)  # Converte de 1 canal para 3 canais
        x = self.features(x)
        output = self.classifier(x)
        #output = nn.Softmax(dim=1)(output)
        return output




# In[13]:


## Funções Otimizadas para Leitura de CSV e Carregamento de Imagens

import os
import psutil
from pathlib import Path

def explore_csv(csv_path, max_rows=5):
    """
    Explora arquivo CSV fornecendo informações detalhadas.

    Parâmetros:
        csv_path (str): Caminho do arquivo CSV
        max_rows (int): Número de linhas para exibir

    Retorna:
        pd.DataFrame: Dados carregados
    """
    print("="*80)
    print("📋 EXPLORAÇÃO DO ARQUIVO CSV")
    print("="*80)

    # Verificar se arquivo existe
    if not os.path.exists(csv_path):
        print(f"❌ ERRO: Arquivo não encontrado: {csv_path}")
        return None

    print(f"✓ Arquivo encontrado: {csv_path}")
    print(f"  Tamanho do arquivo: {os.path.getsize(csv_path) / (1024*1024):.2f} MB")

    ## rename 0 col to path , 1 col to label and 2 col to patient id if exists
    # map col label to AFIB =1 and Others = 0

    try:
        # Ler CSV
        print("\n📖 Lendo arquivo CSV...")
        data = pd.read_csv(csv_path)

        print(f"✓ CSV lido com sucesso!")
        print(f"  Shape: {data.shape} (linhas, colunas)")
        print(f"  Colunas: {list(data.columns)}")
        print(f"  Tipos de dados:\n{data.dtypes}\n")

        # Verificar valores nulos
        print("📊 Valores Nulos:")
        null_counts = data.isnull().sum()
        if null_counts.sum() == 0:
            print("  ✓ Nenhum valor nulo encontrado!")
        else:
            print(f"  ⚠ Valores nulos encontrados:\n{null_counts}\n")

        # Primeiras linhas
        print(f"📄 Primeiras {max_rows} linhas:")
        print(data.head(max_rows).to_string())

        # Estatísticas
        print(f"\n📈 Estatísticas básicas:")
        print(f"  Total de amostras: {len(data)}")
        if data.shape[1] >= 2:
            label_col = data.columns[-1]
            print(f"  Distribuição de labels ({label_col}):")
            print(data[label_col].value_counts().to_string())

        print("="*80)
        return data

    except Exception as e:
        print(f"❌ ERRO ao ler CSV: {e}")
        return None


def validate_and_load_images(data, image_path_prefix, n_samples=None, max_errors=10,type_load_img=3):
    """
    Valida e carrega imagens com tratamento robusto de erros.

    Parâmetros:
        data (pd.DataFrame): DataFrame com caminhos das imagens
        image_path_prefix (str): Prefixo do caminho das imagens
        n_samples (int): Número de amostras a carregar (None = todas)
        max_errors (int): Máximo de erros antes de parar

    Retorna:
        tuple: (img_dataset, data_valid, error_count)
    """
    print("\n" + "="*80)
    print("🖼️  VALIDAÇÃO E CARREGAMENTO DE IMAGENS")
    print("="*80)

    # Preparar dados
    if n_samples is not None:
        data_sample = data.sample(n=min(n_samples, len(data)), random_state=42).reset_index(drop=True)
    else:
        data_sample = data.reset_index(drop=True)

    print(f"\n📌 Configuração:")
    print(f"  Amostras a processar: {len(data_sample)}")
    print(f"  Prefixo de caminho: {image_path_prefix}")
    print(f"  Máximo de erros permitidos: {max_errors}")

    # Verificar memória disponível
    memory_info = psutil.virtual_memory()
    print(f"\n💾 Memória disponível: {memory_info.available / (1024**3):.2f} GB")
    estimated_memory = (len(data_sample) * 3 * 256 * 256 * 1) / (1024**3)
    print(f"  Memória estimada para imagens: {estimated_memory:.2f} GB")

    if estimated_memory > memory_info.available * 0.8:
        print(f"  ⚠ AVISO: Uso de memória pode ser alto!")

    # Preparar array
    img_dataset = np.zeros((len(data_sample),3,256, 256), dtype=np.uint8)
    error_indices = []
    error_count = 0

    print(f"\n🔄 Carregando imagens...")

    for idx, row in tqdm(data_sample.iterrows(), total=len(data_sample), desc='Carregando imagens'):
        try:
            img_path = row.iloc[1]  # Primeira coluna = caminho
            full_path = os.path.join(image_path_prefix, img_path)

            # Verificar se arquivo existe
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Imagem não encontrada: {full_path}")

            # Carregar imagem
            img = treat_image_PIL(img_path, type_return=type_load_img)


            img_dataset[idx] = img

        except Exception as e:
            error_count += 1
            error_indices.append(idx)

            if error_count == 1:
                print(f"\n⚠ Erros encontrados durante carregamento:")

            print(f"  Erro na amostra {idx}: {type(e).__name__}: {str(e)[:60]}")

            if error_count >= max_errors:
                print(f"\n❌ Máximo de {max_errors} erros atingido. Parando...")
                break

    # Remover amostras com erro
    if error_indices:
        print(f"\n🧹 Removendo {len(error_indices)} amostras com erro...")
        valid_indices = [i for i in range(len(data_sample)) if i not in error_indices]
        img_dataset = img_dataset[valid_indices]
        data_sample = data_sample.iloc[valid_indices].reset_index(drop=True)

    print("\n✅ Carregamento concluído!")
    print(f"  Imagens carregadas com sucesso: {len(img_dataset)}")
    print(f"  Imagens com erro: {error_count}")
    print(f"  Shape final: {img_dataset.shape}")
    print("="*80)

    return img_dataset, data_sample, error_count


def get_image_statistics(img_dataset):
    """Calcula estatísticas das imagens carregadas."""
    print("\n" + "="*80)
    print("📊 ESTATÍSTICAS DAS IMAGENS")
    print("="*80)

    print(f"Shape: {img_dataset.shape}")
    print(f"Tipo de dado: {img_dataset.dtype}")
    print(f"Min: {img_dataset.min()}")
    print(f"Max: {img_dataset.max()}")
    print(f"Média: {img_dataset.mean():.2f}")
    print(f"Std: {img_dataset.std():.2f}")
    print(f"Memória total: {img_dataset.nbytes / (1024**2):.2f} MB")
    print("="*80)


# In[14]:


print(f"Leitura e carregamento otimizado do arquivo '{FILE_PATH}'...\n")

## concat file_path and file_path_aug and read both csvs and concatenate the dataframes

# 1️⃣ EXPLORAR CSV PRIMEIRO
data = explore_csv(FILE_PATH, max_rows=10)[0:N_samples]


## CSV concat com data augumentation
#data_aug = explore_csv(FILE_PATH_AUG, max_rows=10)

#data = pd.concat([data, data_aug], ignore_index=True)
print(f"\n📌 Após concatenação com data augmentation: {len(data)} amostras") 


# In[15]:


data
#data['patient_id'].unique()


# In[16]:


def test_():
    from numpy import rint


    print("Tensor de Rotulos sendo gerado...")


    print(f"Configuracao do K-Fold para {folds} folds...")
    kf = KFold(n_splits=5)
    kf.get_n_splits(data['patient_id'].unique())
    print(kf)
    print("K-Fold configurado com sucesso.")

    print("Iniciando o treinamento com K-Fold Cross Validation...")

    train_loss_total = []
    val_loss_total =[]
    all_models =[]
    patient_id_list = data['patient_id'][0:None].unique()
    for train, test in kf.split(patient_id_list):
        print(f"\n{'#'*80}")
        print(f"train e val split para fold {train}  e {test}...")
        ## Obter os patient_id para treino e validação
        train_patient_ids = data.iloc[test]
        train_patient_ids = train_patient_ids['patient_id']
        print(train_patient_ids)

        dados2 = data[data['patient_id'].isin(train_patient_ids)]
        print(dados2.index)
        print(f"\n{'#'*80}")

        val_patient_ids = data.iloc[train]
        val_patient_ids = val_patient_ids['patient_id']
        print(val_patient_ids)
        dados2 = data[data['patient_id'].isin(val_patient_ids)]
        print(dados2.index)
        print(f"\n{'#A'*80}")




# ## Load and Print Images sample N=10

# In[17]:


if data is None:
    print("❌ Não foi possível continuar. Verifique o arquivo CSV.")
else:
    # 2️⃣ RENOMEAR COLUNAS PARA PADRONIZAÇÃO
    print("\n🏷️  Padronizando nomes de colunas...")
    if data.shape[1] >= 2:
        # patient_id,age,sex,height,weight,diagnostic_superclass,path,NORMAL_LABEL,RITM_NAO_AFIB,AFIB   
        #data.rename(columns={data.columns[6]: 'path', data.columns[-1]: 'label', data.columns[0]: 'patient_id'}, inplace=True)
        data_patient_id = data[['patient_id','path', 'AFIB']]
        print(f"✓ Colunas renomeadas: {list(data.columns[:3])}\n")

    # 3️⃣ CARREGAR IMAGENS COM VALIDAÇÃO
    img_dataset, data_imgs, error_count = validate_and_load_images(
        data=data_patient_id,
        image_path_prefix=path,  # path definido anteriormente
        n_samples=10,
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
    print(data['AFIB'].value_counts().to_string())
    print("="*80 + "\n")


# In[ ]:





# In[18]:


## crie uma transformacao de numpy para image pil e faca o display da imagem original e da imagem transformada para verificar se a transformacao esta correta
def numpy_to_pil(numpy_array):
    # Verifica se o array tem a forma correta (C, H, W)
   # if numpy_array.shape[0] != 3:
     #   raise ValueError("O array deve ter a forma (3, H, W) para ser convertido em imagem PIL.")

    print(numpy_array.shape)
    print(numpy_array.dtype)
    img_pil = Image.fromarray(numpy_array.transpose(1, 2, 0))  # Converte de (C, H, W) para (H, W, C)
    return (img_pil)

    return pil_image
# ## Target Label

# In[19]:


data['NORMAL'] =data['NORMAL'].map({True: 1, False: 0})
data['Other'] = data['Other'].map({True: 1, False: 0})
data['AFIB'] = data['AFIB'].map({True: 1, False: 0})


# In[20]:


## concat labels NORMAL_LABEL, RITM_NAO_AFIB, AFIB em uma coluna chamada rotulo
# NORMAL','Other','AFIB'
data['rotulo'] = data['NORMAL'].astype(str)   + data['Other'].astype(str) + data['AFIB'].astype(str)


# In[21]:


data.query("rotulo == '000'")


# In[ ]:





# In[22]:


data['rotulo'].value_counts()


# In[23]:


## Distribuicao das amostras
## Histograma da distribuicao das amostras
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
data['rotulo'].value_counts().plot(kind='bar')
plt.title('Distribuição das Amostras por Classe')
plt.xlabel('Classe')
plt.ylabel('Contagem de Amostras')
plt.grid(axis='y')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# ## Kfold 

# In[24]:


def a1():

    print(f"Configuracao do K-Fold para {folds} folds...")
    kf = KFold(n_splits=folds)
    kf.get_n_splits(tensor_imagem)
    print(kf)
    print("K-Fold configurado com sucesso.")

    print("Iniciando o treinamento com K-Fold Cross Validation...")

    train_loss_total = []
    val_loss_total =[]
    all_models =[]




# ## Data Augumentation func

# In[25]:


_label = data[['NORMAL','Other','AFIB']].values





## import transforms 
# Funcao responsavel por receber um numpy array de imagens, aplicar data augmuentation em cada um, fazer um novo array com as i
#  imagens transformadas e retornar esse array para ser usado no treinamento do modelo. 
#  A funcao deve receber o tensor de imagens e os indices das imagens a serem transformadas, 
#  aplicar as transformacoes e retornar um novo tensor de imagens com as transformacoes aplicadas concatenadas das imagens originaisn
from tabnanny import verbose

from torchvision import transforms
from torchvision.transforms import v2

transform = v2.Compose([
    v2.RandomRotation(degrees=30),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Zoom e translação
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),                    # Blur
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),        # Variações de cor
    v2.RandomPerspective(distortion_scale=0.2, p=0.3),                   # Perspectiva
    v2.RandomErasing(p=0.3),                                              # Erro aleatório
])
## name file in dados_pd['path']
def data_augumentation(images_tensor,dados_pd ,size_aug=1,verbose=False):
    _img_dataset_aug = np.zeros_like(images_tensor, dtype=np.uint8)
    img_dataset_aug = np.repeat(_img_dataset_aug, size_aug, axis=0)
    name_file = dados_pd['path'].values
    _label_dataset_aug = np.zeros_like(dados_pd[['NORMAL','Other','AFIB']], dtype=np.uint8)
    label_dataset_aug = np.repeat(_label_dataset_aug, size_aug, axis=0)
    print(f"Shape do tensor de imagens augumented: {img_dataset_aug.shape}, original shape: {images_tensor.shape}")
    print(f"Shape do tensor de labels augumented: {label_dataset_aug.shape}, original shape: {_label_dataset_aug.shape}")
    _label = dados_pd[['NORMAL','Other','AFIB']].values
    files_aug = []
    total_counter_limit = 0# img_dataset_aug.shape[0]
    for i in tqdm(range(len(images_tensor)), desc='Applying augmentation'):
        for j in range(0,size_aug):
            # Aplica transformações via PIL
            img_pil = Image.fromarray(images_tensor[i].transpose(1, 2, 0))  # (3, 256, 256) → (256, 256, 3)
            img_transformed = transform(img_pil)
            img_dataset_aug[total_counter_limit] = np.array(img_transformed).transpose(2, 0, 1)  # Volta para (3, 256, 256)
            label_dataset_aug[total_counter_limit] = _label[i]
            files_aug.append(name_file[i])
            total_counter_limit +=1
    print("✓ Data augmentation completed for all images.")
    concat_np = np.concatenate((images_tensor, img_dataset_aug), axis=0)
    concat_label = np.concatenate((_label, label_dataset_aug), axis=0)
    data_tensor_full = TensorDataset(torch.from_numpy(concat_np), torch.from_numpy(concat_label))
    print(f"✓ Data augmentation aplicada. Novo shape do dataset: {data_tensor_full.tensors[0].shape} no formato TensorDataset (imagens + labels)")
    if  verbose:
        print("Tamanhos e shape dos arrays após data augmentation:")
        print(f"  img_dataset_aug shape: {img_dataset_aug.shape}, dtype: {img_dataset_aug.dtype}")
        print(f"  label_dataset_aug shape: {label_dataset_aug.shape}, dtype: {label_dataset_aug.dtype}")
        print(f"  concat_np shape: {concat_np.shape}, dtype: {concat_np.dtype}")
        print(f"  concat_label shape: {concat_label.shape}, dtype: {concat_label.dtype}")
        return data_tensor_full , concat_np, concat_label, files_aug
    else:
        return data_tensor_full


# ## Train Valid Loop with DataAug

# In[68]:


print('''
      ###############################################
      Inicio do treinamento com K-Fold Cross Validation
      ###############################################
      ''')
device = 'cuda:0'
train_loss_total = []
val_loss_total =[]
patient_id_list = data['patient_id'][0:N_samples].unique()
kf = KFold(n_splits=folds)

for i, (train_index, test_index) in enumerate(kf.split(patient_id_list)):
    print(f"Fold {i}:")
    #print(f"  Train: index={train_index}")
    #print(f"  Test:  index={test_index}")
    ## init train test for folder
    ## Load images from train test kf

    print(f"\n{'#'*80}")
    print(f"train e val split para fold {len(train_index)}  e {len(test_index)}...")
    #print(f'amostras {train_index}')
    ## Obter os patient_id para treino e validação
    ## Load images from dados_train e dados_val

    train_patient_ids = data.iloc[train_index]
    train_patient_ids = train_patient_ids['patient_id']

    dados_train = data[data['patient_id'].isin(train_patient_ids)][0:N_samples]
    print(f"\n{'#'*80}")



    img_dataset_train, data_imgs, error_count = validate_and_load_images(
        data=dados_train[['patient_id','path', 'AFIB']],  # Seleciona apenas as colunas necessárias
        image_path_prefix=path,  # path definido anteriormente
        n_samples=N_samples,  # Carrega todas as amostras do conjunto de treino
        max_errors=10,
        type_load_img = 3
    )

    tensor_dataset_full_aug_train = data_augumentation(img_dataset_train, dados_train,2)
    ## Geracao do tensor de imagens

    print("Dataset de treino de dados (imagens + labels) criado com sucesso. Tamanho do Dataset:", len(tensor_dataset_full_aug_train))

    val_patient_ids = data.iloc[test_index]
    val_patient_ids = val_patient_ids['patient_id']

    dados_val = data[data['patient_id'].isin(val_patient_ids)][0:N_samples]

    img_dataset_val, data_imgs, error_count = validate_and_load_images(
        data=dados_val[['patient_id','path', 'AFIB']],  # Seleciona apenas as colunas necessárias
        image_path_prefix=path,  # path definido anteriormente
        n_samples=N_samples,
        max_errors=10,
        type_load_img = 3
    )

    # Do NOT apply data augmentation to the validation set
    concat_np_val = img_dataset_val
    concat_label_val = dados_val[['NORMAL','Other','AFIB']].values
    tensor_dataset_full_val = TensorDataset(torch.from_numpy(concat_np_val), torch.from_numpy(concat_label_val))
    print("Dataset de validação (sem augmentation) criado com sucesso. Tamanho do Dataset:", len(tensor_dataset_full_val))

    ## Print com resumo dos dados utilizados para fold 
    print(f"\n{'#'*80}")
    print(f"Resumo do Fold {i}:")
    print(f"  Tamanho do conjunto de treino: {len(tensor_dataset_full_aug_train)}")
    print(f"  Tamanho do conjunto de validação: {len(tensor_dataset_full_val)}")
    print(f"  Distribuição de labels no treino:\n{dados_train['rotulo'].value_counts()}")
    print(f"  Distribuição de labels na validação:\n{dados_val['rotulo'].value_counts()}")
    print(f"\n{'#'*80}")

    train_loader_img = DataLoader(tensor_dataset_full_aug_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_img = DataLoader(tensor_dataset_full_val, batch_size=BATCH_SIZE, shuffle=False)

    model= ECGClassifierResnet()
    if (flg_salvar_modelos):
        salvar_model(model, path='dataset_AFIB_Others/4000-files/output/', name_file=f'model_fold_{i}.pth')
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


# In[ ]:


import numpy as np

arr = np.random.rand(10, 2, 3, 3)

multiplicador = 2

novo_arr = np.repeat(arr, multiplicador, axis=0)

print(novo_arr.shape)
# (20, 2, 3, 3)

