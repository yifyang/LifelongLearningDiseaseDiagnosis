import random
import torch
import argparse
from statistics import mean
from data_process import split_data, gen_data
from train import train
from evaluate import evaluate_model
from utils import get_sent_embed, get_word_embed
from alignment_model import update_word_alignment_model, update_sent_alignment_model


def random_cluster(num_clusters, num_diseases, cluster_disease = None):
    each_cluster = int(num_diseases / num_clusters)
    cluster_index = list(range(1, num_diseases+1))
    random.shuffle(cluster_index)
    cluster_labels = {}

    if cluster_disease is None:
        cluster_disease = {}
        for i in range(num_clusters):
            cluster_disease[i] = []
            for j in range(i*each_cluster, (i+1)*each_cluster):
                cluster_disease[i].append(cluster_index[j])
                cluster_labels[cluster_index[j]] = i
    else:
        for k, v in cluster_disease.items():
            for label in v:
                cluster_labels[label] = k

    return cluster_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model details
    parser.add_argument('--n_hiddens', type=int, default=200)
    parser.add_argument('--n_embeds', type=int, default=300)
    parser.add_argument('--n_tasks', type=int, default=10)
    parser.add_argument('--n_outputs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--memory_strength', type=float, default=0)
    parser.add_argument('--sent_memory_size', type=int, default=256)
    parser.add_argument('--word_memory_size', type=int, default=256)
    parser.add_argument('--fix_cluster', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2020)

    # Alignment Model details
    parser.add_argument('--am_sent_epoch', type=int, default=20)
    parser.add_argument('--am_word_epoch', type=int, default=30)
    parser.add_argument('--am_word_lr', type=float, default=0.00002)
    parser.add_argument('--am_sent_lr', type=float, default=0.0001)
    parser.add_argument('--am_sent_batch', type=int, default=32)
    parser.add_argument('--am_word_batch', type=int, default=32)
    parser.add_argument('--am_hiddens', type=int, default=200)
    parser.add_argument('--am_embed_save', type=str, default='embedding/')
    parser.add_argument('--am_mode', type=int, default=1)

    # experiment parameters
    parser.add_argument('--cuda', type=str, default='no',
                        help='Use GPU?')
    parser.add_argument('--cuda_sel', type=int, default=0,
                        help='select GPU')
    parser.add_argument('--device', type=str, default='cpu',
                        help='GPU or CPU')

    # data parameters
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--train_file', default='train_data_v3-5.txt',
                        help='data file')
    parser.add_argument('--test_file', default='test_data_v3-5.txt',
                        help='data file')
    parser.add_argument('--fasttext_file', default='cc.zh.300.vec',
                        help='data file')
    args = parser.parse_args()

    print("Loading and preprocessing data...")
    train_path = args.data_path + args.train_file
    test_path = args.data_path + args.test_file
    fasttext_path = args.data_path + args.fasttext_file
    args.device = torch.device('cuda:'+str(args.cuda_sel) if torch.cuda.is_available() else 'cpu')

    if args.fix_cluster:
        # fix type of diseases in each task
        cluster_disease = {0: [24, 34, 38, 40], 1: [22, 29, 33, 39], 2: [1, 12, 13, 19],
                           3: [2, 5, 31, 17], 4: [3, 6, 8, 20], 5: [4, 7, 9, 30],
                           6: [10, 11, 23, 32], 7: [14, 25, 26, 35], 8: [15, 18, 28, 37],
                           9: [16, 21, 27, 36]}
    else:
        cluster_disease = None

    cluster_label = random_cluster(args.n_tasks, args.n_outputs, cluster_disease=cluster_disease)

    train_data, test_data, vocabulary, embedding = \
        gen_data(train_path, test_path, args, fasttext_path)

    split_train = split_data(train_data, args.n_tasks, cluster_label)
    split_test = split_data(test_data, args.n_tasks, cluster_label)
    print("Loading and preprocessing done")

    memory_sent_data = []
    memory_word_data = []
    memory_sent_embed = []
    memory_word_embed = []
    save_word_embed = []
    cur_model = None
    word_alignment_model = None
    sent_alignment_model = None
    results = []
    random.seed(args.seed)

    for t in range(args.n_tasks):
        print("Training on task {}...".format(t+1))
        cur_train_data = split_train[t]
        cur_test_data = split_test[t]

        cur_model = train(cur_train_data, vocabulary, embedding,
                          args, cur_model, memory_sent_data,
                          word_alignment_model, sent_alignment_model)

        memory_sent_data.append(cur_train_data[-args.sent_memory_size:])
        memory_word_data.append(cur_train_data[-args.word_memory_size:])
        memory_sent_embed.append(get_sent_embed(cur_model, memory_sent_data[-1],
                                                args.batch_size, args.device,
                                                sent_alignment_model))
        memory_word_embed.append(get_word_embed(cur_model, memory_word_data[-1],
                                                args.batch_size, args.device,
                                                word_alignment_model))

        print("Training on task {} end".format(t+1))

        if len(memory_sent_data) > 1 and args.am_mode:
            print("Training of the alignment model...")

            # Get embeddings of the memory from previous tasks
            # Input of the alignment model training
            cur_sent_embed = [get_sent_embed(cur_model, this_memory,
                                             args.batch_size, args.device,
                                             sent_alignment_model, True)
                              for this_memory in
                              memory_sent_data]
            cur_word_embed = [get_word_embed(cur_model, this_memory,
                                             args.batch_size, args.device,
                                             word_alignment_model, True)
                              for this_memory in
                              memory_word_data]

            sent_alignment_model = update_sent_alignment_model(sent_alignment_model, cur_sent_embed,
                                                          memory_sent_embed, args)
            word_alignment_model = update_word_alignment_model(word_alignment_model, cur_word_embed,
                                                          memory_word_embed, args)

            # Get the target of the alignment model training
            memory_sent_embed = [get_sent_embed(cur_model, this_memory,
                                                args.batch_size, args.device,
                                                sent_alignment_model, False)
                             for this_memory in
                             memory_sent_data]
            memory_word_embed = [get_word_embed(cur_model, this_memory,
                                                args.batch_size, args.device,
                                                word_alignment_model, False)
                                 for this_memory in
                                 memory_word_data]

            print("Training of the alignment model done")

        print("Evaluation start")
        results = [evaluate_model(cur_model, split_test[tt], args.batch_size, args.device,
                   word_alignment_model, sent_alignment_model) for tt in range(0, t+1)]
        print(results)
        print("Average accuracy: ", mean(results))
