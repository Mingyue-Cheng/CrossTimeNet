import torch
import warnings
import os
from args import args
from datautils import load_ETT
warnings.filterwarnings("ignore")
from dataset import Dataset, ForecastingDataset
from data_provider.data_factory import data_provider
from Bert4ts import Bert4ts
from Bert4ts_1 import Bert4ts_1
from GPT4ts import GPT4ts
import torch.utils.data as Data
from forecasting import Bert4tsForecasting
from pretrain import Bert4tsPretrain

def get_data(flag, data=None):
        # print(data)
        data_set, data_loader = data_provider(args, flag, data)
        return data_set, data_loader

def load_train_data():
    """
    根据是否打乱数据的设置加载训练数据。
    """
    model_dict = {
         'Bert':Bert4ts,
         'gpt':GPT4ts,
         'bert':Bert4ts_1
    }
    data_prefix = '/data/tinyy/CrossTimeNet/dataset'
    vqvae_model_prefix = '/data/tinyy/CrossTimeNet/256_64_7/tokenizer/'
    model = model_dict[args.model]()
    # model.init_vqvae(args.vqvae_model_path)

    train_loaders = []

    if args.shuffle:
        for datapath in ['ETTh1', 'ETTm1', 'ETTh2', 'ETTm2','weather','exchange_rate']:
            vqvae_model_path = vqvae_model_prefix+datapath+'/model.pkl'
            print(vqvae_model_path)
            model.init_vqvae(vqvae_model_path)
            args.data_path=datapath+".csv"
            args.data = datapath
            args.root_path=data_prefix
            # data_path = os.path.join(data_prefix, datapath)
            data_set, train_loader = get_data("train")
            # Train_data_all, Train_data, VAL_data, Test_data = load_ETT(data_path+ '/')
            # train_dataset = Dataset(
            # args=args,
            # data=Train_data_all
            # )
            # train_loader = Data.DataLoader(
            # train_dataset, batch_size=args.train_batch_size, shuffle=True
            # )
            train_loaders.append(train_loader)
        return model, train_loaders
    else:
        # Train_data_all, Train_data, VAL_data, Test_data = load_ETT(args.data_path)
        # train_dataset = Dataset(
        # args=args,
        # data=Train_data_all
        # )
        # train_loader = Data.DataLoader(
        # train_dataset, batch_size=args.train_batch_size, shuffle=True
        # )
        data_set, train_loader = get_data("train")
        model.init_vqvae(args.vqvae_model_path)
        return model, train_loader

def pretrain():
    torch.set_num_threads(12)
    torch.cuda.manual_seed(args.seed)

    model, train_loader = load_train_data()
    # args.data_shape = train_loader[0].dataset.shape() if args.shuffle else train_loader.dataset.shape()

    # print(args.data_shape)
    print("Dataset initialization complete.")

    trainer = Bert4tsPretrain(args, model, train_loader)
    trainer.pretrain()

if __name__ == "__main__":
    pretrain()
