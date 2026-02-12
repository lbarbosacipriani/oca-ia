import torch
from torch import nn
import numpy as np
import os
import torch.optim as optim
from tqdm import tqdm
from lib.output_metrics import create_folder_if_not_exists


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

def salvar_model(model, path = 'output/modelos', name_file='model.pth'):
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

def simple_loop(model, train_image, val_image, epochs, batch_size, fold_index):
    # Simple training loop
    num_epochs = epochs
    train_losses, val_losses = [], []
    lim_loss = 1.5
    iter_size = batch_size
    print(f'Number of training images per iteration: {iter_size}')
    #model = modelo( num_classes=5)
  # model.to(device)
    #criterion =  nn.NLLLoss()
    criterion =  nn.BCEWithLogitsLoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    predict_label_full_train = []
    predict_label_full = []
    true_label_full_train = []
    true_label_full = []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        predict_label_train =[]
        true_label_train =[]
        running_loss_train = 0.0
        for images, labels in tqdm(train_image, desc='Training loop'):
            # Move inputs and labels to the device
            images = images.to(torch.float)
            image, label = images, labels
            optimizer.zero_grad()
            outputs = model(image)
            loss_train = criterion(outputs.float(), label.float())

            #loss_train = criterion(outputs, label)
            loss_train.backward()
            optimizer.step()
            running_loss_train += loss_train.item() * label.size(0)
            try:
                _pred_train = outputs.cpu().data.numpy().astype(int).T[0].tolist()
                if(len(_pred_train) == iter_size):
                    predict_label_train.append(_pred_train)
                    _true_train = label.cpu().data.numpy().astype(int).T[0].tolist()
                    true_label_train.append(_true_train)

            except Exception as e:
                print(f"Concatenation error iter: {e}")
                print(_pred_train)
                print(predict_label_train)    
        train_loss = running_loss_train / len(train_image.dataset)
        train_losses.append(train_loss)
        try:
            _p_train = predict_label_train
            _t_train = true_label_train
            predict_label_full_train.append(_p_train)
            true_label_full_train.append(_t_train)
        except Exception as e:
            print(f"Concatenation error full: {e}")
            print(predict_label_train)
            print(true_label_train)
        model.eval()
        running_loss_valid = 0.0
        rotulos =[] 
        predict_label =[]
        true_label =[]
        _iter=0
        with torch.no_grad():
            for images, labels in tqdm(val_image, desc='Validation loop'):
                # Move inputs and labels to the device
                images = images.to(torch.float)
                images, label = images, labels
                rotulos.append(label.cpu().data.numpy())
                outputs = model(images)

                loss_valid = criterion(outputs.float(), label.float())
                #loss_valid = criterion(outputs, label)

                #print( [outputs.cpu().data.numpy().astype(int).T[0]])
                #print(label.cpu().data.numpy().astype(int).T[0])
                #print(predict_label)
                try:
                    _pred = outputs.cpu().data.numpy().astype(int).T[0].tolist()

                    if(len(_pred) == iter_size):
                        predict_label.append(_pred)
                        _true = label.cpu().data.numpy().astype(int).T[0].tolist()
                        true_label.append(_true)
                except Exception as e:
                    print(f"Concatenation error iter: {e}")
                    print(_pred)
                    print(predict_label)    
                running_loss_valid += loss_valid.item() * label.size(0)
                _iter +=1
        val_loss = running_loss_valid / len(val_image.dataset)
        val_losses.append(val_loss)
        print(f'End validation for epoch {epoch}')
        print(f'Amount of images validated: {val_image}')
        print(f'Label predict shape : {len(predict_label)} for epoch {epoch}')
        print(f'Count of iterations: {_iter} for epoch {epoch}')
        try:
            _p = predict_label
            _t = true_label
            predict_label_full.append(_p)
            true_label_full.append(_t)
        except Exception as e:
            print(f"Concatenation error full: {e}")
            print(predict_label)
            print(true_label)
        #val_acc = accuracy_score(rotulos,output_model)
        print(f'Val accuracy {epoch}:')
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

    predict_label_full_out = np.array(predict_label_full)
    true_label_full_out = np.array(true_label_full)
    predict_label_full_train_out = np.array(predict_label_full_train)
    true_label_full_train_out = np.array(true_label_full_train)
    print(f'Salvado das metricas de validacao e treino')
    salvar_metricas(path=f'output/metricas/valid/fold_{fold_index}', name_file_train=f'predict_label_valid_fold_{fold_index}.npy', name_file_val=f'true_label_valid_fold_{fold_index}.npy', predict_label= predict_label_full_out, true_label= true_label_full_out)
    salvar_metricas(path=f'output/metricas/train/fold_{fold_index}', name_file_train=f'predict_label_train_fold_{fold_index}.npy', name_file_val=f'true_label_train_fold_{fold_index}.npy', predict_label= predict_label_full_train_out, true_label= true_label_full_train_out)
    print(f'Finalizado o salvamento das metricas')
    return train_losses, val_losses, model,predict_label_full_out, true_label_full_out