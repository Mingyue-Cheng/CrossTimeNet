import torch
import numpy as np
from tqdm import tqdm
from torch import optim
import os
from torch.cuda.amp import autocast, GradScaler
import transformers
import random
from args import args


class Bert4tsPretrain:
    def __init__(self, args, model, data_loader):
        self.args = args
        self.train_loader = data_loader
        self.model = model
        self.num_epochs = args.pre_num_epoch
        self.results_dir = args.save_path
    
    

    def pretrain(self):
        # 初始化迭代器和混合索引
        if self.args.shuffle:
            train_loader_etth1, train_loader_ettm1, train_loader_etth2,train_loader_ettm2, train_loader_weather,train_loader_exchange= self.train_loader
            data_iters = {
                0: iter(train_loader_etth1),
                1: iter(train_loader_ettm1),
                2: iter(train_loader_etth2),
                3: iter(train_loader_ettm2),
                4: iter(train_loader_weather),
                5:iter(train_loader_exchange),
            }
            lengths = [len(loader) for loader in self.train_loader]
            r = [i for i, l in enumerate(lengths) for _ in range(l)]
            random.shuffle(r)
            step_len = len(r)
        else:
            step_len = len(self.train_loader)

        # 配置优化器和学习率调度器
        param_dict = [{"params": self.model.parameters(), "lr": self.args.lr}]
        optimizer = optim.Adam(param_dict, weight_decay=1e-5)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=2000,
            num_training_steps=self.num_epochs * step_len,
        )
        scaler = GradScaler()
        best_loss = float('inf')

        for p in self.model.parameters():
            p.requires_grad = True

        for epoch in range(self.num_epochs):
            self.model.train()
            step, train_losses = 0, 0.0
            if self.args.shuffle:
                random.shuffle(r)
                tqdm_iter = tqdm(r, desc=f"GPT Epoch {epoch+1}", ncols=120)
                data_iters = {
                0: iter(train_loader_etth1),
                1: iter(train_loader_ettm1),
                2: iter(train_loader_etth2),
                3: iter(train_loader_ettm2),
                4: iter(train_loader_weather),
                5:  iter(train_loader_exchange),
                }
                for i in tqdm_iter:
                    try:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = next(data_iters[i])
                        # inputs, labels = next(data_iters[i])
                    except StopIteration:
                        data_iters[i] = iter(self.train_loader[i])
                        batch_x, batch_y, batch_x_mark, batch_y_mark = next(data_iters[i])
                        # inputs, labels = next(data_iters[i])
                    # batch_x, batch_y, batch_x_mark, batch_y_mark = next(data_iters[i])
                    loss_value = self._train_step(batch_x,batch_y, optimizer, scaler, scheduler,i)
                    train_losses += loss_value
                    step += 1
                    tqdm_iter.set_postfix({"loss": format(train_losses / step, ".4f")})
            else:
                tqdm_iter = tqdm(self.train_loader, desc=f"GPT Epoch {epoch+1}", ncols=120)
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm_iter):
                    loss_value = self._train_step(batch_x, batch_y, optimizer, scaler, scheduler)
                    train_losses += loss_value
                    step += 1
                    tqdm_iter.set_postfix({"loss": format(train_losses / step, ".4f")})

            avg_train_loss = format(train_losses / step, ".4f")
            print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss}")

            self._save_checkpoint(epoch, optimizer, avg_train_loss, best_loss)
            if train_losses / step < best_loss:
                best_loss = train_losses / step
                self._save_checkpoint(epoch, optimizer, avg_train_loss, best_loss, best=True)

    def _train_step(self, inputs, labels, optimizer, scaler, scheduler,dataid=None):
        with autocast():
            outputs = self.model(inputs.float().to(args.device), pretrain=True,dataid=dataid)
        scaler.scale(outputs.loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        return outputs.loss.cpu().item()

    def _save_checkpoint(self, epoch, optimizer, avg_train_loss, best_loss, best=False):
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": avg_train_loss,
        }
        if best:
            model_save_path = os.path.join(self.results_dir, "best_model.pth")
            print(f"Best model saved to {model_save_path}")
        else:
            model_save_path = os.path.join(self.results_dir, "latest_checkpoint.pth")
            print(f"Latest model saved to {model_save_path}")
        torch.save(checkpoint, model_save_path)

        # 保存训练损失到文本文件
        loss_save_path = os.path.join(self.results_dir, "train_losses.txt")
        with open(loss_save_path, "a") as f:
            f.write(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}\n")
