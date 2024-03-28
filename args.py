import argparse
import os
import json

parser = argparse.ArgumentParser(description="Input hyperparams.")
parser.add_argument("--seed", type=int, default=66)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--data_path", type=str, default="ecg-data")
parser.add_argument("--local_model_path", type=str, default="bert-base-uncased")
parser.add_argument("--vqvae_model_path", type=str, default="har_64_128_4")
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=16)
parser.add_argument("--results_dir", type=str, default="experiment_results_vqvae_eeg")
parser.add_argument("--sub_dir", type=str, default="linear_evaluation")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--train_batch_size_pretrain", type=int, default=32)
parser.add_argument("--learning_rate_pretrain", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--pretrain_num_epochs", type=int, default=30)
parser.add_argument("--pretrain_mlp_num_epochs", type=int, default=20)
parser.add_argument("--frozen", type=int, default=False, help="If use frozen gpt2, linear")
parser.add_argument("--embeding", type=str, default="word_mapping", help="If use frozen gpt2, linear")
parser.add_argument("--wave_length", type=int, default=50)
parser.add_argument("--n_embed", type=int, default=512)
parser.add_argument("--class_num", type=int, default=27)
parser.add_argument("--seq_len", type=int, default=5000)
parser.add_argument("--feat_dim", type=int, default=12)
parser.add_argument("--params", type=int, default=False)
parser.add_argument("--classifier", type=int, default=True)
parser.add_argument("--fine_tuning_all", type=int, default=False)
parser.add_argument("--pretrain_stage", type=int, default=False)
parser.add_argument("--load_pretrain", type=int, default=False)
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--wordmapping", type=str, default="0")
parser.add_argument("--train_ratio", type=float, default=1)
parser.add_argument("--mask_ratio", type=float, default=0.3)



args = parser.parse_args()


# if args.pretrain_one:
#     args.train_ratio = 1000000

if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)



full_path = args.results_dir + '/'+args.sub_dir
if not os.path.exists(full_path):
    os.mkdir(full_path)

config_file = open(full_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()