import torch.nn as nn
import torch.nn.functional as F
import time
import math
import datetime
import arguments
import os

def get_train_data(input_file):
        rows_list = []
        # Using readlines()
        file1 = open(input_file, 'r',encoding = "ISO-8859-1")
        Lines = file1.readlines()
        # Strips the newline character
        for line in Lines:
            #count += 1
            line_bk = line.split(" ", 1)
            rows_list.append((line_bk[1].strip('\n'),line_bk[0].replace(":", "_")))
        return rows_list

def get_label_list():
    label_list = ["UNK_UNK","ABBR_abb", "ABBR_exp", "DESC_def", "DESC_desc",
              "DESC_manner", "DESC_reason", "ENTY_animal", "ENTY_body",
              "ENTY_color", "ENTY_cremat", "ENTY_currency", "ENTY_dismed",
              "ENTY_event", "ENTY_food", "ENTY_instru", "ENTY_lang",
              "ENTY_letter", "ENTY_other", "ENTY_plant", "ENTY_product",
              "ENTY_religion", "ENTY_sport", "ENTY_substance", "ENTY_symbol",
              "ENTY_techmeth", "ENTY_termeq", "ENTY_veh", "ENTY_word", "HUM_desc",
              "HUM_gr", "HUM_ind", "HUM_title", "LOC_city", "LOC_country",
              "LOC_mount", "LOC_other", "LOC_state", "NUM_code", "NUM_count",
              "NUM_date", "NUM_dist", "NUM_money", "NUM_ord", "NUM_other",
              "NUM_perc", "NUM_period", "NUM_speed", "NUM_temp", "NUM_volsize",
              "NUM_weight"]
    return label_list


def get_qc_examples(input_file):
  """Creates examples for the training and dev sets."""
  examples = []

  with open(input_file, 'r') as f:
      contents = f.read()
      file_as_list = contents.splitlines()
      for line in file_as_list[1:]:
          split = line.split(" ")
          question = ' '.join(split[1:])

          text_a = question
          inn_split = split[0].split(":")
          label = inn_split[0] + "_" + inn_split[1]
          examples.append((text_a, label))
      f.close()

  return examples

def print_training_stats(training_stats):
    for stat in training_stats:
        print(stat)


# device
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def save_checkpoint(model, is_final, path, filename, checkpoint='checkpoint'):
    if is_final:
        filepath = os.path.join(path, filename) 
        torch.save(model.state_dict(), filepath+'.pt')
    else:
        filepath = os.path.join(checkpoint, filename)
        torch.save(model.state_dict(), filepath+'.pt')
        #shutil.copyfile(filepath, os.path.join(checkpoint,'model_best.pth.tar'))

def print_train_perfomance(path,filename):

    for stat in training_stats:
        print(stat)
    
def print_args():
    args = arguments.get_args()
    print(args)

def print_tree(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

