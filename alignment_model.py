import torch
import torch.nn as nn
import random
import numpy as np

torch.manual_seed(1)

class AlignmentModel(nn.Module):
    """
    General alignment model
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super(AlignmentModel, self).__init__()
        if hidden_dim is not None:
            # init for word_alignment_model
            self.mode = "word"
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, output_dim)
        else:
            # init for sentence_alignment_model
            self.mode = "sent"
            self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.mode == "sent":
            out = self.linear(x)
        else:
            hidden_state1 = self.linear1(x)
            out = self.linear2(hidden_state1)
        return out

def process_samples(cur_que_embed, memory_que_embed):
    total_cur_embed = cur_que_embed[0]
    total_mem_embed = memory_que_embed[0]

    for i in range(1, len(cur_que_embed)):
        total_cur_embed = np.concatenate((total_cur_embed, cur_que_embed[i]))
        total_mem_embed = np.concatenate((total_mem_embed, memory_que_embed[i]))

    index = list(range(len(total_cur_embed)))
    random.shuffle(index)

    return total_cur_embed[index], total_mem_embed[index]


def feed_samples(alignment_model, x_train, y_correct, criterion, optimiser, device, batch_size):
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_correct).to(device)
    optimiser.zero_grad()
    for i in range((len(x_train)-1)//batch_size+1):
        batch_inputs = inputs[i*batch_size:(i+1)*batch_size]
        batch_labels = labels[i*batch_size:(i+1)*batch_size]
        batch_outputs = alignment_model.forward(batch_inputs)
        loss = criterion(batch_outputs, batch_labels)
        loss.backward()
        optimiser.step()

def update_word_alignment_model(alignment_model, cur_word_embed, memory_word_embed, args):
    if alignment_model is None:
        alignment_model = AlignmentModel(args.n_hiddens*2, args.n_hiddens*2, args.am_hiddens)
    alignment_model = alignment_model.to(args.device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(alignment_model.parameters(), lr = args.am_word_lr)
    rand_cur_embed, rand_mem_embed = process_samples(cur_word_embed, memory_word_embed)
    am_word_epoch = args.am_word_epoch
    for epoch in range(am_word_epoch):
        feed_samples(alignment_model, rand_cur_embed,
                     rand_mem_embed, criterion,
                     optimiser, args.device, args.am_word_batch)

    return alignment_model

def update_sent_alignment_model(alignment_model, cur_sent_embed, memory_sent_embed, args):
    if alignment_model is None:
        alignment_model = AlignmentModel(args.n_hiddens*2, args.n_hiddens*2)
    alignment_model = alignment_model.to(args.device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(alignment_model.parameters(), lr = args.am_sent_lr)
    rand_cur_embed, rand_mem_embed = process_samples(cur_sent_embed, memory_sent_embed)
    for epoch in range(args.am_sent_epoch):
        feed_samples(alignment_model, rand_cur_embed,
                     rand_mem_embed, criterion,
                     optimiser, args.device, args.am_sent_batch)

    return alignment_model
