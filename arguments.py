import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    parser.add_argument('--datasetsrc', type=str, default='huggingface', help=' dataset source.') 
    parser.add_argument('--dataset', type=str, default='trec', help='Name of the dataset used.') 
    parser.add_argument('--input_path', type=str, default='./data_source', help='training datapath if it is on local ')
    parser.add_argument('--label_pct', type=float, default=0.5, help='percentage of labelled data for training.')
    parser.add_argument('--model_name', type=str, default='bert-base-cased', help='pretrained bert model used.')
    parser.add_argument('--max_seq_length', type=int, default=64, help='Transformer parameters')   
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size used for training and testing')
    parser.add_argument('--hidden_sizes', type=list, default=[512], help='list of Hidden size ')
    parser.add_argument('--input_size', type=int, default=512, help='Discriminator input size ')
    parser.add_argument('--output_size', type=int, default=512, help='Generator output size ')
    parser.add_argument('--num_hidden_layers_g', type=int, default=1, help='num_hidden_layers of generator')
    parser.add_argument('--num_hidden_layers_d', type=int, default=1, help='num_hidden_layers of discriminator') 
    parser.add_argument('--noise_size', type=int, default=100, help='noise_size')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='generator loss noise: epsilon')
    parser.add_argument('--out_dropout_rate', type=float, default=0.2, help='drop out rate')
    parser.add_argument('--apply_balance', type=bool, default=True, help='balance for sampling')
    parser.add_argument('--multi_gpu', type=bool, default=True, help='multi gpu')
    parser.add_argument('--apply_scheduler', type=bool, default=False, help='scheduler for training')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='warmup proportion.')
    parser.add_argument('--print_each_n_step', type=int, default=10, help='print each_n_step')
    
    parser.add_argument('--train_epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--learning_rate_discriminator', type=int, default=5e-5, help='Discriminator learning rate')
    parser.add_argument('--learning_rate_generator', type=int, default=5e-5, help='Generator learning rate')    
    
    parser.add_argument('--test_filename', type=str, default='./ganbert/data/test.tsv', help='Path to where the test file data')
    parser.add_argument('--train_out_path', type=str, default='./train', help='training output path')
    parser.add_argument('--train_log_name', type=str, default='train_log.csv', help='training statistics log file name' )

    ### args.train_flag:
    parser.add_argument('--train_flag', type=bool, default=False, help='train  model or not')
    ###save models 
    parser.add_argument('--save_models_flag', type=bool, default=False, help='save trained models or not')
   # parser.add_argument('--trained_model_path', type=str, default='./modelpath', help='training model path')
    parser.add_argument('--generator_file', type=str, default='generator', help='generator model file name to be saved')
    parser.add_argument('--discriminator1_file', type=str, default='discriminator1', help='classifier/discriminator1 model file name to be saved')
    parser.add_argument('--discriminator3_file', type=str, default='discriminator3', help='bi-discriminator model file name to be saved')
#     parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')
#     parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
#     parser.add_argument('--log_name', type=str, default='accuracies.log', help='Final performance of the models will be saved with this name')
    parser.add_argument('--component', type=str, default='perf', help='choose the component to run: train model-train, summarize performance-perf, query')
    args = parser.parse_args()

    #if not os.path.exists(args.out_path):
     #   os.mkdir(args.out_path)
    
    return args
