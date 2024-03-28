import os, csv
import torch
import pickle
import scipy.signal as signal
import torch.nn as nn
import warnings
import torch.nn.functional as Fd
from collections import Counter
from sklearn.metrics import *

import transformers
from torch.cuda.amp import autocast, GradScaler
from tqdm.notebook import tqdm
from bert4timeseries import bert4ts
from data import Dataset
from torch.utils.data import DataLoader
from torch import optim
import os
import numpy as np
from tqdm import tqdm
from args import args
import random
from sklearn.metrics import f1_score
import random
device = 'cuda:0'


def Pretrain(train_loader, model, results_dir):
    train_dataset_ecg,train_dataset_eeg,train_dataset_har = train_loader
    data_iter_ecg = iter(train_dataset_ecg)
    data_iter_eeg = iter(train_dataset_eeg)
    data_iter_har = iter(train_dataset_har)
    l_ecg, l_eeg, l_har = [len(i) for i in train_loader]
    r = []
    r = r + [0]*l_ecg
    r = r + [1]*l_eeg
    r = r + [2]*l_har
    random.shuffle(r)
    num_epochs = args.pretrain_num_epochs
    param_dict = [
        {"params": model.parameters(), "lr": args.learning_rate_pretrain},
    ]
    optimizer = optim.Adam(param_dict, weight_decay=1e-5)
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=2000,
        num_training_steps=num_epochs * len(r),
    )
    scaler = GradScaler()

    best = 0.0
    best_loss = 10000000000000000000

    for p in model.parameters():
        p.requires_grad = True
    
    total_step = 0
    for epoch in range(num_epochs):
        model.train()
        step, train_losses = 0, 0.0
        random.shuffle(r)
        tqdm_iter = tqdm(r, desc=f"GPT Epoch {epoch+1}", ncols=120)
        data_iter_ecg = iter(train_dataset_ecg)
        data_iter_eeg = iter(train_dataset_eeg)
        data_iter_har = iter(train_dataset_har)
        
        for i in tqdm_iter:
            total_step += 1
            if i==0: inputs, labels = next(data_iter_ecg)
            if i==1: inputs, labels = next(data_iter_eeg)
            if i==2: inputs, labels = next(data_iter_har)
            #print(inputs.shape)
            #continue
            with autocast():
                outputs = model(inputs, pretrain=True,dataid=i)
                
            scaler.scale(outputs.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            loss_value = outputs.loss.cpu().item()
            train_losses += loss_value
            step += 1
            tqdm_iter.set_postfix({"loss": format(train_losses / step, ".4f")})

        avg_train_loss = format(train_losses / step, ".4f")
        print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}')

        if train_losses / step < best_loss:
            best_loss = train_losses / step
            model_save_path = os.path.join(results_dir, 'best_model.pth')  # Change this to your desired file path
            checkpoint = {
                'epoch': epoch + 1,  # Save the current epoch
                'model_state_dict': model.state_dict(),  # Save the model's state dictionary
                'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer's state dictionary
                'best_loss': avg_train_loss  # Save the best validation accuracy
            }
            #model.save_pretrained(model_save_path)
            torch.save(checkpoint, model_save_path)
            print(f'Best model saved to {model_save_path}')
        checkpoint = {
            'epoch': epoch + 1,  # Save the current epoch
            'model_state_dict': model.state_dict(),  # Save the model's state dictionary
            'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer's state dictionary
            'best_loss': avg_train_loss  # Save the best validation accuracy
        }
        model_save_path = os.path.join(results_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, model_save_path)
        print(f'latest model saved to {model_save_path}')    
        loss_save_path = os.path.join(results_dir, 'train_losses.txt')
        with open(loss_save_path, 'a') as f:
            f.write(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}\n')

def classifier_train(train_loader, test_loader, num_epochs, model, results_dir, pretrain,dataid): #是否进行预训练
    
    
    criterion = nn.CrossEntropyLoss()
    param_dict = [
        {"params": model.parameters(), "lr": args.learning_rate},
    ]
    optimizer = optim.Adam(param_dict, weight_decay=1e-5)
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=2000,
        num_training_steps=num_epochs * len(train_loader),
    )
    scaler = GradScaler()

    
    checkpoint_path = os.path.join(results_dir, 'best_model.pth')
    if pretrain:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0)
        start_epoch = checkpoint['epoch']
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded. Starting from epoch {start_epoch}.")
    else:
        start_epoch = 0
        best_val_accuracy = -1000000  # Initialize if no checkpoint is found
        print("No checkpoint found. Starting from scratch.")

    results_dir = results_dir + '/'+  args.sub_dir
    os.makedirs(results_dir, exist_ok=True)

    for p in model.parameters():
        p.requires_grad = True

    best_val_accuracy = -1000000
    for epoch in range(num_epochs):
        model.train()
        step, train_losses = 0, 0.0
        tqdm_iter = tqdm(train_loader, desc=f"GPT Epoch {epoch+1}", ncols=120)
        for inputs, labels in tqdm_iter:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            #print(f'label:{labels.shape}',inputs.shape)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs.float(),dataid=dataid)
                #print(outputs.shape)
                loss = criterion(outputs, labels.long())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses += loss.item()
            step += 1
            tqdm_iter.set_postfix({"loss": format(train_losses / step, ".4f")})

        avg_train_loss = format(train_losses / step, ".4f")
        print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}')

        # 保存训练损失到文本文件	    
        loss_save_path = os.path.join(results_dir, 'train_losses.txt')
        with open(loss_save_path, 'a') as f:
            f.write(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}\n')

        # Evaluation loop
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        threshold = 0.5  # 设定阈值
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs.float(),dataid=dataid)
                loss = criterion(outputs, labels.long())
                total_val_loss += loss.item()
                predicted = outputs.max(dim=-1)[1]
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = format(total_val_loss / step, ".4f")
        val_accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}%')

        loss_save_path = os.path.join(results_dir, 'train_losses.txt')
        with open(loss_save_path, 'a') as f:
            f.write(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}%\n')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            # Save the current best model
            model_save_path = os.path.join(results_dir, 'best_model.pth')  # Change this to your desired file path
            checkpoint = {
                'epoch': epoch + 1,  # Save the current epoch
                'model_state_dict': model.state_dict(),  # Save the model's state dictionary
                'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer's state dictionary
                'best_val_accuracy': best_val_accuracy  # Save the best validation accuracy
            }

            torch.save(checkpoint, model_save_path)
            print(f'Best model saved to {model_save_path}')

        model_save_path = os.path.join(results_dir, 'latest_checkpoint.pth')  # Fixed file name for the latest checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'scheduler_state_dict': scheduler.state_dict()  # Save the state of the scheduler as well, if necessary
        }

        torch.save(checkpoint, model_save_path)
        print(f'Latest checkpoint saved at epoch {epoch+1} to {model_save_path}')

        scheduler.step()

def classifier_test(results_dir, test_loader, model,dataid):
    results_dir = results_dir + '/'+ args.sub_dir
    os.makedirs(results_dir, exist_ok=True)
    model_save_path = os.path.join(results_dir, 'best_model.pth')
    checkpoint = torch.load(model_save_path)  # Replace with the correct file path

    # Load the model's state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(args.device)
    criterion = nn.CrossEntropyLoss()
    print("evaluation loop")
    # Evaluation loop
    model.eval()
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    total_val_loss = 0.0
    threshold = 0.5  # 设定阈值

    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs,dataid=dataid)
            loss = criterion(outputs, labels.long())
            total_val_loss += loss.item()

            predicted = outputs.max(dim=-1)[1]

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_val_loss = total_val_loss / len(test_loader)
    accuracy = 100. * correct / total

    recall = recall_score(y_true=all_labels, y_pred=all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')  # 你可以选择不同的average参数


    # 保存F1分数到文件
    with open(os.path.join(results_dir, 'test_results.txt'), 'w') as f:
        f.write(f'Test Accuracy: {accuracy}%\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'recall Score: {recall}\n')

# Set random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def main():

    # 初始化模型、损失函数和优化器
    seed_everything(args.seed)
    pretrain = args.pretrain_stage
    embeding = args.embeding
    model = bert4ts(embeding).to(args.device)

    print('model inital.')
    All_Train,All_Test = [],[]
    for data_path in ['ecg-data','eeg-data','har-data']:

        #vqvae_model_path = args.data_path + '/' + args.vqvae_model_path + '/model.pkl'
        vqvae_model_path = args.data_path  + '/model.pkl'
        print(vqvae_model_path)
        model.init_vqvae(vqvae_model_path)
        print('model inital.')

        test_path = data_path + '/samples_test.pkl'
        train_path = data_path + '/samples_train.pkl'
        #加载数据
        with open(train_path, 'rb') as file:
            samples_train = pickle.load(file)
        with open(test_path, 'rb') as file:
            samples_test = pickle.load(file)
        ecg_train = []
        label_train = []
        ecg_test = []
        label_test = []

        text, _, vector = samples_train[0]

        print(text)
        print(vector)

        for _, ecg, label in samples_train[:int(len(samples_train)*min(args.train_ratio,1))]:
            label_train.append(label)
            ecg_train.append(ecg)
        for _, ecg, label in samples_test:
            label_test.append(label)
            ecg_test.append(ecg)

        Train_data, Test_data = [ecg_train, label_train], [ecg_test, label_test]
        print(len(label_train), len(ecg_test))
        All_Train.append(Train_data)
        All_Test.append(Test_data)
        

    # 示例使用的参数

    # 初始化数据集对象
    train_dataset_ecg = Dataset(args.device, 'train', All_Train[0], All_Test[0])
    train_dataset_eeg = Dataset(args.device, 'train', All_Train[1], All_Test[1])
    train_dataset_har = Dataset(args.device, 'train', All_Train[2], All_Test[2])

    test_dataset_ecg = Dataset(args.device, 'test', All_Train[0], All_Test[0])
    test_dataset_eeg = Dataset(args.device, 'test', All_Train[1], All_Test[1])
    test_dataset_har = Dataset(args.device, 'test', All_Train[2], All_Test[2])
    # 创建数据加载器
    train_loader_ecg = DataLoader(train_dataset_ecg, batch_size=args.train_batch_size, shuffle=True)
    train_loader_eeg = DataLoader(train_dataset_eeg, batch_size=args.train_batch_size, shuffle=True)
    train_loader_har = DataLoader(train_dataset_har, batch_size=args.train_batch_size, shuffle=True)

    test_loader_ecg = DataLoader(test_dataset_ecg, batch_size=args.test_batch_size, shuffle=False)
    test_loader_eeg = DataLoader(test_dataset_eeg, batch_size=args.test_batch_size, shuffle=False)
    test_loader_har = DataLoader(test_dataset_har, batch_size=args.test_batch_size, shuffle=False)
    print('data inital.')

    os.makedirs(args.results_dir, exist_ok=True)
    #model pretrain
    if pretrain:
        print('pretrain begin.')
        Pretrain(train_loader=[train_loader_ecg,train_loader_eeg,train_loader_har], model=model, results_dir=args.results_dir)
    if args.data_path == 'ecg-data':
        train_loader = train_loader_ecg
        test_loader  = test_loader_ecg
        dataid = 0
    if args.data_path == 'eeg-data':
        train_loader = train_loader_eeg
        test_loader  = test_loader_eeg
        dataid = 1
    if args.data_path == 'har-data':
        train_loader = train_loader_har
        test_loader  = test_loader_har
        dataid = 2
    if args.classifier:
        print("classifrier train begin.")
        classifier_train(train_loader=train_loader, test_loader=test_loader, num_epochs=args.num_epochs, results_dir=args.results_dir, pretrain=args.load_pretrain, model=model,dataid=dataid)
        print("classifier test begin.")
        classifier_test(results_dir=args.results_dir, test_loader=test_loader, model=model,dataid=dataid)
    else:
        print("pretrain end.")


if __name__ == '__main__':
    main()