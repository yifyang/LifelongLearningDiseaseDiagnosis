import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from model import MultiModel
from utils import ranking_sequence

def process_samples(sample_list, device):
    sentences = []
    targets = []
    all_words = []
    for sample in sample_list:
        sentence = torch.tensor(sample[1], dtype=torch.long).to(device)
        words = torch.tensor(sample[2], dtype=torch.long).to(device)
        targets.append(sample[0]-1)
        sentences += [sentence]
        all_words.append(words)

    targets = torch.tensor(targets, dtype=torch.long).to(device)
    return sentences, targets, all_words

def feed_samples(model, samples, loss_function, device,
                 word_alignment_model=None, sent_alignment_model=None):
    sentences, targets, words = process_samples(samples, device)

    # rank sentences with lengths
    ranked_sentences, alignment_sentence_indexs = ranking_sequence(sentences)
    ranked_words, alignment_words_indexs = ranking_sequence(words)
    sentence_lengths = [len(sentence) for sentence in ranked_sentences]
    words_lengths = [len(word) for word in ranked_words]

    # Padding
    pad_sentences = torch.nn.utils.rnn.pad_sequence(ranked_sentences)
    pad_sentences = pad_sentences.to(device)
    pad_words = torch.nn.utils.rnn.pad_sequence(ranked_words)
    pad_words = pad_words.to(device)

    model.zero_grad()
    if word_alignment_model is not None:
        word_alignment_model.zero_grad()
        sent_alignment_model.zero_grad()
    output = model(pad_sentences, pad_words,
                   alignment_sentence_indexs, alignment_words_indexs,
                   sentence_lengths, words_lengths,
                   word_alignment_model, sent_alignment_model)

    loss = loss_function(output, targets)
    loss.backward()
    return loss

def train(training_data, vocabulary, embedding, args, model=None,
          memory_data=[], word_alignment_model=None,
          sent_alignment_model=None):
    """
    main training procedure

    :return: the updated model
    """
    # init the model
    if model is None:
        torch.manual_seed(100)
        model = MultiModel(len(vocabulary), np.array(embedding), args)
    loss_function = nn.CrossEntropyLoss()
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    memory_index = 0
    for epoch_i in range(args.epoch):
        for i in range((len(training_data)-1)//args.batch_size+1):
            samples = training_data[i*args.batch_size:(i+1)*args.batch_size]

            if len(memory_data) > 0:
                # replay previous tasks with memory
                memory_batch = memory_data[memory_index]
                loss = feed_samples(model, memory_batch,
                                    loss_function, args.device,
                                    word_alignment_model,
                                    sent_alignment_model)
                optimizer.step()
                memory_index = (memory_index+1) % len(memory_data)

            loss = feed_samples(model, samples, loss_function,
                                args.device, word_alignment_model,
                                sent_alignment_model)

            optimizer.step()
            del loss
    return model