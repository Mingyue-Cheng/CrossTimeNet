import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import time


class Bert4tsForecasting:
    def __init__(self, args, model, data_loader):
        self.args = args
        self.device = args.device
        self.model = model
        self.train_loader, self.val_loader, self.test_loader = data_loader
        self.pred_len = args.pred_len
        self.verbose = True
        self.lr_decay = 0.98
        self.num_epoch = args.num_epoch
        self.channels = args.feat_dim
        self.samples = self.pred_len * self.channels
        self.save_path = args.save_path_each_pred

    def forecasting_finetune(self):
        self.result_file = open(self.save_path + "/train_result.txt", "w")
        self.result_file.close()
        self.result_file = open(self.save_path + "/test_result.txt", "w")
        self.result_file.close()

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: self.lr_decay**step,
            verbose=self.verbose,
        )
        self.criterion = torch.nn.MSELoss()

        self.best_val_loss = 10000

        for epoch in range(self.num_epoch // 10):
            train_loss_epoch, train_time_cost = self._train_single_epoch()
            val_loss_epoch, val_time_cost = self._eval_single_epoch()
            curr_lr = self.scheduler.get_last_lr()[0]
            print(f"Current learning rate: {curr_lr}")
            self.scheduler.step()

            self.result_file = open(self.save_path + "train_result.txt", "a+")
            self.print_process(
                "Finetune epoch:{0},loss:{1},training_time:{2}".format(
                    epoch + 1, train_loss_epoch, train_time_cost
                )
            )
            print("epoch:{0},Current learning rate: {1}".format(epoch+1, curr_lr), file=self.result_file)
            print(
                "Finetune train epoch:{0},loss:{1},training_time:{2}".format(
                    epoch + 1, train_loss_epoch, train_time_cost
                ),
                file=self.result_file,
            )
            self.print_process(
                "Finetune epoch:{0},loss:{1},validation_time:{2}".format(
                    epoch + 1, val_loss_epoch, val_time_cost
                )
            )
            print(
                "Finetune epoch:{0},loss:{1},validation_time:{2}".format(
                    epoch + 1, val_loss_epoch, val_time_cost
                ),
                file=self.result_file,
            )
            self.result_file.close()

            if val_loss_epoch < self.best_val_loss:
                self.best_val_loss = val_loss_epoch
                torch.save(
                    self.model.state_dict(), self.save_path + "finetune_model.pkl"
                )

        test_loss, mae = self._eval_model()
        self.result_file = open(self.save_path + "test_result.txt", "a+")
        print(f"Test loss: {test_loss}", file=self.result_file)
        print(f"Mae: {mae}", file=self.result_file)
        self.result_file.close()

        self.print_process("Test loss: {0}".format(test_loss))
        self.print_process("Mae: {0}".format(mae))
        return test_loss, mae

    def _train_single_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader) if self.verbose else self.train_loader

        loss_sum = 0
        for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm_dataloader):
            batch_y = batch_y.float().to(self.device)
            batch_x = batch_x.float().to(self.device)

            self.optimizer.zero_grad()
            
            pred_output = self.model(batch_x, pretrain=False)
            f_dim = -1 if self.args.features == 'MS' else 0
            pred_output = pred_output[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss = self.criterion(pred_output, batch_y)
            loss_sum += loss.item()
            loss.backward()
            self.optimizer.step()
        t1 = time.perf_counter()

        return loss_sum / len(self.train_loader), t1 - t0

    def _eval_single_epoch(self):
        t0 = time.perf_counter()
        self.model.eval()
        tqdm_dataloader = tqdm(self.val_loader) if self.verbose else self.val_loader

        loss_sum = 0

        with torch.no_grad():
            for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm_dataloader):
                batch_y = batch_y.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                

                pred_output = self.model(batch_x, pretrain=False)
                f_dim = -1 if self.args.features == 'MS' else 0
                pred_output = pred_output[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = self.criterion(pred_output, batch_y)
                loss_sum += loss.item()

        t1 = time.perf_counter()
        return loss_sum / len(self.val_loader), t1 - t0

    def _eval_model(self):
        self.model.load_state_dict(torch.load(self.save_path + "finetune_model.pkl"))
        self.model.eval()
        tqdm_dataloader = tqdm(self.test_loader) if self.verbose else self.test_loader

        loss_sum = 0
        mae_sum = 0
        total_samples = 0
        with torch.no_grad():
            for idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm_dataloader):
                batch_y = batch_y.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                

                pred_output = self.model(batch_x, pretrain=False)
                f_dim = -1 if self.args.features == 'MS' else 0
                pred_output = pred_output[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = self.criterion(pred_output, batch_y)
                loss_sum += loss.item()# 计算 MAE
                mae = torch.mean(torch.abs(pred_output - batch_y))
                mae_sum += mae.item() * batch_y.size(0)
                total_samples += batch_y.size(0)

        return loss_sum / len(self.test_loader) , mae_sum / total_samples

    def print_process(self, *x):
        if self.verbose:
            print(*x)
