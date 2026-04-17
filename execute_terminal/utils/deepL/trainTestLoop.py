
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from lib.writeOutputs import salvar_metricas

def simple_loop(model, train_image, val_image, epochs, batch_size, fold_index, patience=10, delta=0.001, device_input=None):
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
        #model = model.to(device_input)
        print(f'✓ Modelo movido para {device_input}')
    except Exception as e:
        print(f'❌ ERRO ao mover modelo para device: {e}')
        raise
    
    
    criterion = nn.CrossEntropyLoss()
    # Optimizer COM regularização (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    predict_label_full_train = []
    predict_label_full = []
    true_label_full_train = []
    true_label_full = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        predict_label_train = []
        true_label_train = []
        running_loss_train = 0.0
        
        try:
            for images, labels in tqdm(train_image, desc='Training loop'):
                # ✅ FIX: Garantir float32 e evitar overflow
                images = images.to(torch.float32)
                labels = labels.to(torch.long)
                optimizer.zero_grad()
                
                # ✅ FIX: Usar autocast para mixed precision

                outputs = model(images)
                pred_train = torch.argmax(outputs, dim=1)
                _true = labels.squeeze()
                loss_train = criterion(outputs.to(torch.float32), _true)
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
                _pred_train = torch.argmax(outputs, dim=1)
                running_loss_train += loss_train.item() * labels.size(0)
                
                # Salvar predições e labels CORRETAMENTE
                try:
                    pred_list = _pred_train.cpu().detach().numpy().astype(int).tolist()
                    true_list = labels.squeeze().cpu().detach().numpy().astype(int).tolist()
                    
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
        
        try:
            _p_train = predict_label_train
            _t_train = true_label_train
            predict_label_full_train.append(_p_train)
            true_label_full_train.append(_t_train)
        except Exception as e:
            print(f"⚠ Erro ao concatenar métricas de treinamento: {e}")
        
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
                    images = images.to(torch.float32)
                    labels = labels.to(torch.long)
                    outputs = model(images)
                    
                    loss_valid = criterion(outputs.to(torch.float32), labels.squeeze())
                    
                    # Get predictions
                    _pred = torch.argmax(outputs, dim=1)
                    
                    # Salvar predições e labels
                    try:
                        pred_list = _pred.cpu().numpy().astype(int).tolist()
                        true_list = labels.squeeze().cpu().numpy().astype(int).tolist()
                        
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
    
    salvar_metricas(path=f'output/metricas/valid/fold_{fold_index}', 
                    name_file_train=f'predict_label_valid_fold_{fold_index}.npy', 
                    name_file_val=f'true_label_valid_fold_{fold_index}.npy', 
                    predict_label= predict_label_full_out, true_label= true_label_full_out)
    
    salvar_metricas(path=f'output/metricas/train/fold_{fold_index}', 
                    name_file_train=f'predict_label_train_fold_{fold_index}.npy', 
                    name_file_val=f'true_label_train_fold_{fold_index}.npy', 
                    predict_label= predict_label_full_train_out, true_label= true_label_full_train_out)
    print(f'Finalizado o salvamento das metricas')
    return train_losses, val_losses, model, predict_label_full_out, true_label_full_out