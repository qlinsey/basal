import pandas as pd
import csv
from gutils import *
from dbbuilder import *
from model import *
from basal import *
import torch
import io
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import datetime
import torch.nn as nn
from transformers import *
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import arguments
import sys, getopt
import warnings
from pretrain_clustering import *
from utils import *


warnings.filterwarnings("ignore")
pd.options.display.max_columns = None
pd.options.display.max_rows = None


def train_test_split(df, n):
    train = df.sample(n=n, axis=0)
    test = df.drop(index=train.index)
    return train, test


def main(args):
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    if args.datasetsrc == "huggingface":
        model_name = args.model_name
        transformer = AutoModel.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        label_pct = args.label_pct
        train_ds, test_ds = load_dataset(args.dataset, split=["train", "test"])
        train_df = pd.DataFrame(train_ds)
        train_df["label"] = train_df["fine_label"]
        print("train_df.columns=", train_df.columns)
        len_train = int(len(train_df) * args.label_pct)
        test_df = pd.DataFrame(test_ds)
        test_df["label"] = test_df["fine_label"]
        label_list = np.unique(train_df["label"].values)
        print("No. of unique labels=", len(label_list))

        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        labeled_df, unlabeled_df = train_test_split(train_df, len_train)

        labeled_examples = list(zip(labeled_df["text"], labeled_df["label"]))
        unlabeled_examples = list(zip(unlabeled_df["text"], unlabeled_df["label"]))
        test_examples = list(zip(test_df["text"], unlabeled_df["label"]))

        # ------------------------------
        #   Load the train dataset
        # ------------------------------
        train_examples = labeled_examples
        train_label_masks = np.ones(len(labeled_examples), dtype=bool)

        tmp_masks = np.zeros(len(unlabeled_examples), dtype=bool)

        train_dataloader = generate_data_loader(
            train_examples,
            tokenizer,
            train_label_masks,
            label_map,
            args.max_seq_length,
            args.batch_size,
            do_shuffle=True,
            balance_label_examples=args.apply_balance,
        )

        unlabeled_dataloader = generate_data_loader(
            unlabeled_examples,
            tokenizer,
            tmp_masks,
            label_map,
            args.max_seq_length,
            args.batch_size,
            do_shuffle=True,
            balance_label_examples=args.apply_balance,
        )

        # ------------------------------
        #   Load the test dataset
        # ------------------------------
        test_label_masks = np.ones(len(test_examples), dtype=bool)
        test_dataloader = generate_data_loader(
            test_examples,
            tokenizer,
            test_label_masks,
            label_map,
            args.max_seq_length,
            args.batch_size,
            do_shuffle=False,
            balance_label_examples=False,
        )

    config = AutoConfig.from_pretrained(args.model_name)
    hidden_size = int(config.hidden_size)
    # Define the number and width of hidden layers
    hidden_levels_g = [hidden_size for i in range(0, args.num_hidden_layers_g)]
    hidden_levels_d = [hidden_size for i in range(0, args.num_hidden_layers_d)]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    # -------------------------------------------------
    #   Instantiate the Generator and Discriminator
    # -------------------------------------------------
    generator = Generator(
        noise_size=args.noise_size,
        output_size=hidden_size,
        hidden_sizes=hidden_levels_g,
        dropout_rate=args.out_dropout_rate,
    )
    discriminator = Discriminator(
        input_size=hidden_size,
        hidden_sizes=hidden_levels_d,
        num_labels=len(label_list),
        dropout_rate=args.out_dropout_rate,
    )
    discriminator3 = BiDiscriminator(
        input_size=hidden_size,
        hidden_sizes=hidden_levels_d,
        num_labels=2,
        dropout_rate=args.out_dropout_rate,
    )

    #  GPU if available
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        discriminator3.cuda()
        transformer.cuda()
        if args.multi_gpu:
            transformer = torch.nn.DataParallel(transformer)
    if args.train_flag:
        print("training...")
        gmal = Basaler(args, device, label_list)
        (
            val_accuracy,
            generator,
            transformer,
            discriminator,
            discriminator3,
        ) = gmal.train(
            train_dataloader,
            unlabeled_dataloader,
            test_dataloader,
            transformer,
            generator,
            discriminator,
            discriminator3,
        )
    else:
        print("scoring...")
        predict(test_examples, label_map, args.max_seq_length, args.batch_size)
    if args.save_models_flag:
        save_checkpoint(
            generator,
            True,
            args.train_out_path,
            args.generator_file,
            checkpoint="checkpoint",
        )
        save_checkpoint(
            dsicriminator1,
            True,
            args.train_out_path,
            args.discriminator1_file,
            checkpoint="checkpoint",
        )
        save_checkpoint(
            discriminator3,
            True,
            args.train_out_path,
            args.discriminator3_file,
            checkpoint="checkpoint",
        )


def query_label(args, num_labels, k):
    model_name = "bert-base-cased"
    config = AutoConfig.from_pretrained(model_name)
    hidden_size = int(config.hidden_size)
    hidden_levels_d = [hidden_size for i in range(0, args.num_hidden_layers_d)]
    discriminator = Discriminator(
        input_size=hidden_size,
        hidden_sizes=hidden_levels_d,
        num_labels=num_labels,
        dropout_rate=args.out_dropout_rate,
    )
    unlabel_preds = []
    un_labels_ids = []
    pre_label_probs = []
    pre_labels = []
    k = 10
    real_batch = 20  # 52
    cnt = 0
    total_test_loss = 0
    nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for cnt, batch in enumerate(unlabeled_dataloader):  # ):
        # print("cnt=",cnt,"batch[0].shape=",batch[0].shape[0])
        if batch[0].shape[0] != 32:
            continue
        t_input_ids = batch[0][0:real_batch].to(device)
        t_input_mask = batch[1][0:real_batch].to(device)
        t_labels = batch[2][0:real_batch].to(device)
        with torch.no_grad():
            model_outputs = transformer(t_input_ids, attention_mask=t_input_mask)
            hidden_states = model_outputs[-1]
            _, logits, probs = discriminator(hidden_states)
            ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
            filtered_logits = logits[:, 0:-1]
            # Accumulate the test loss.
            total_test_loss += nll_loss(filtered_logits, t_labels)

        _, preds = torch.max(filtered_logits, 1)
        pre_label_idx = torch.argmax(F.softmax(filtered_logits, dim=-1), dim=1)
        pre_label_prob, _ = torch.max(F.softmax(filtered_logits, dim=-1), dim=1)
        pre_label_probs += pre_label_prob
        pre_label = [label_list[i] for i in pre_label_idx.detach().cpu().numpy()]
        pre_labels += pre_label
        unlabel_preds += preds.detach().cpu()
        un_labels_ids += t_labels.detach().cpu()

    # Report the final accuracy for this validation run.
    unlabel_preds_c = torch.stack(unlabel_preds).numpy()
    un_labels_ids = torch.stack(un_labels_ids).numpy()
    test_accuracy = np.sum(unlabel_preds_c == un_labels_ids) / len(unlabel_preds)
    print("  Accuracy: {0:.3f}".format(test_accuracy))
    print("unlabel_preds_c=", unlabel_preds_c[0:10])
    print("un_labels_ids=", un_labels_ids[0:10])

    topk_ids = np.argsort(pre_label_probs)[-k:]
    topk_prob = [pre_label_probs[i].detach().cpu().numpy() for i in topk_ids]
    topk_pre_labels = un_labels_ids[topk_ids]
    topk_data_point = [unlabeled_examples[i] for i in topk_ids]


def predict(test_examples, label_map, max_seq_length, batch_size):
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    model_name = "bert-base-cased"
    # transformer = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    hidden_size = int(config.hidden_size)
    hidden_levels_d = [hidden_size for i in range(0, args.num_hidden_layers_d)]
    predictor = torch.load("./saved_models/model-50/ganbart-full-discrminator-50.pt")
    transformer = torch.load("./saved_models/model-50/ganbart-full-transformer-50.pt")
    # predictor = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d, num_labels=len(label_list), dropout_rate=args.out_dropout_rate).cuda()
    # predictor.load_state_dict(torch.load("./saved_models/model-50/ganbart-discrminator.pt"))
    predictor.eval()
    # transformer.load_state_dict(torch.load("./saved_models/ganbart-transformer.pt"))
    # transformer.eval()
    test_label_masks = np.ones(len(test_examples), dtype=bool)
    test_dataloader = generate_data_loader(
        test_examples,
        tokenizer,
        test_label_masks,
        label_map,
        max_seq_length,
        batch_size,
        do_shuffle=False,
        balance_label_examples=False,
    )
    real_batch = 20
    test_preds = []
    test_labels_ids = []
    test_idexes = []
    test_label_probs = []
    test_labels = []
    for cnt, batch in enumerate(test_dataloader):
        t_index = batch[0][0:real_batch].to(device)
        t_input_ids = batch[1][0:real_batch].to(device)
        t_input_mask = batch[2][0:real_batch].to(device)
        t_labels = batch[3][0:real_batch].to(device)

        with torch.no_grad():
            model_outputs = transformer(t_input_ids, attention_mask=t_input_mask)
            hidden_states = model_outputs[-1]
            _, logits, probs = predictor(hidden_states)
            ###log_probs = F.log_softmax(probs[:,1:], dim=-1)
            filtered_logits = logits[:, 0:-1]

            _, preds = torch.max(filtered_logits, 1)
            pre_label_idx = torch.argmax(F.softmax(filtered_logits, dim=-1), dim=1)
            pre_label_prob, _ = torch.max(F.softmax(filtered_logits, dim=-1), dim=1)
            test_label_probs += pre_label_prob.detach().cpu().numpy().tolist()
            test_label = [
                label_list[i] for i in pre_label_idx.detach().cpu().numpy().tolist()
            ]
            test_labels += test_label
            test_preds += preds.detach().cpu()
            test_labels_ids += t_labels.detach().cpu()
            test_idexes += t_index.detach().cpu().numpy().tolist()
            # print("t_index.detach().cpu()=",t_index.detach().cpu().numpy())
    test_data = [test_examples[i] for i in test_idexes]
    df_pre = pd.DataFrame(
        {
            "index": test_idexes,
            "predicted_prob": test_label_probs,
            "predicted_label": test_labels,
            "data_point": test_data,
        }
    )
    print(df_pre.head(5))


def retrain():
    print(
        "Mode 1: percentage of correct predicted labels collected from expert; Mode 2: previous vs. current model accuracy"
    )
    modes = int(input("evaluate from mode_1 or mode_2, please enter 1 or 2:"))
    print("scoring ...")
    print("evaluting ...")
    if modes == 1:
        da_df = pd.read_csv("./label/expert_done/predicted_labelled.csv")
        print("total labelled data: ", da_df.shape[0])
        no_correct = len(np.where(da_df["predicted_label"] == da_df["expert_label"])[0])
        print("no. of predicted==labell data: ", no_correct)
        pct_correct = no_correct / da_df.shape[0]
        print("correct percentage :", pct_correct, ">=0.5: ", pct_correct >= 0.5)
        if pct_correct >= 0.5:
            print("NO more labeling required.")
        else:
            print("Need more experts to label data.")
    else:
        ### mode 2:
        df = pd.read_csv("./report/qc_fine_50_report.csv")
        df = df.drop(columns=["Unnamed: 0"])
        df = df.rename(
            columns={
                "Training Loss generator": "Train_Loss_G",
                "Training Loss multi-discriminator": "Train_Loss_M_D",
                "Training Loss bi-discriminator": "Train_Loss_Bi_D",
            }
        )
        df2 = pd.read_csv("./report/qc_fine_20_report.csv")
        pre_acc = df2["Valid. Accur."].iloc[-1]
        curr_acc = df["Valid. Accur."].iloc[-1]
        print(
            "current model accuracy: ",
            curr_acc,
            "and delta accuracy: ",
            abs(curr_acc - pre_acc),
        )
        if abs(curr_acc - pre_acc) <= 0.01:
            print("NO more labeling required.")
        else:
            print("Need more experts to label data.")


if __name__ == "__main__":
    args = arguments.get_args()
    print("Running component: ", args.component)
    print("Current dataset: ", args.dataset)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    if args.component == "perf":
        pct = int(
            input("20% labelled  or 50% labelled performance? please enter 20 or 50: ")
        )
        print("Fetching performance report for...")
        if pct == 50:
            df = pd.read_csv("./report/qc_fine_50_report.csv")
        elif pct == 20:
            df = pd.read_csv("./report/qc_fine_20_report.csv")
        df = df.drop(columns=["Unnamed: 0"])
        df = df.rename(
            columns={
                "Training Loss generator": "Train_Loss_G",
                "Training Loss multi-discriminator": "Train_Loss_M_D",
                "Training Loss bi-discriminator": "Train_Loss_Bi_D",
            }
        )
        print(df)
    elif args.component == "query":
        option = int(
            input(
                "Which sampling strategy: Random - 1; LeastConfident - 2; MaxMargin - 3; MaxEntropy -4? Please enter your choice:  "
            )
        )
        print("select k data points to be labelled...")
        k = int(input("enter number of data points to be labelled: "))
        if option == 1:
            rows_list = get_train_data("./data_source/qcfine/train_5500.label.txt")
            df_l = pd.DataFrame(rows_list)
            print(df_l[0][:k])
        else:
            df_k = pd.read_csv("./label/inprogress/top-" + str(k) + "-label.csv")
            df_k = df_k.drop(columns=["Unnamed: 0"])
            print(df_k)
    elif args.component == "info":
        ds = [d for d in os.listdir(args.input_path) if not os.path.isdir(d)]
        # ds = os.listdir(args.input_path)
        print("data directories=", ds)
    elif args.component == "tree":
        print_tree("./")
    elif args.component == "score":
        args.train_flag = False
        # location = str(input("enter data points to be predicted:
        print("Fetching data points to be predicted from ./label/to_predict ...\n")
        main(args)
    elif args.component == "retrain":
        retrain()
    elif args.component == "coldstart":
        category = str(input("which category (eid)?"))
        eid = "54419c41f3d2d41d660e3a5bb90571aacfef6faa35220d90af296a0a457e231e"
        initial_data = data_input("./data_source/subject_lines/", eid)
        print("total data points =", len(initial_data))

        option = int(
            input(
                "How many data points to be selected for initial labeling? Please enter your choice: 10% -1 or 20% -2: "
            )
        )
        ratio = 0.2
        if option == 1:
            ratio = 0.1
        embeddings_st2 = data_embedding("", initial_data)
        output_data = run_cluster(
            len(initial_data) * ratio, initial_data, embeddings_st2, eid
        )
        print("out=", output_data.head(2))
    else:
        main(args)
