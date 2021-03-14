import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        self.hidden = self.build_hidden()

    def build_hidden(self, batch_size = 1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return [torch.zeros(2, batch_size, self.hidden_dim),
                torch.zeros(2, batch_size, self.hidden_dim)]

    def init_hidden(self, device='cpu', batch_size = 1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden = (torch.zeros(2, batch_size, self.hidden_dim).to(device),
                torch.zeros(2, batch_size, self.hidden_dim).to(device))

    def forward(self, packed_embeds):
        lstm_out, self.hidden = self.lstm(packed_embeds, self.hidden)
        permuted_hidden = self.hidden[0].permute([1,0,2]).contiguous()
        return permuted_hidden.view(-1, self.hidden_dim*2), lstm_out

class SentenceModel(nn.Module):
    """
    Model of context model
    """
    def __init__(self, vocab_size, vocab_embedding, args):
        super(SentenceModel, self).__init__()
        self.batch_size = args.batch_size
        self.n_hiddens = args.n_hiddens
        self.sentence_biLstm = BiLSTM(args.n_embeds, args.n_hiddens, args.batch_size)

    def init_hidden(self, device, batch_size=1):
        self.sentence_biLstm.init_hidden(device, batch_size)

    def init_embedding(self, vocab_embedding):
        self.word_embeddings.weight.data.copy_(torch.from_numpy(vocab_embedding))

    # def ranking_sequence(self, sequence):
    #     word_lengths = torch.tensor([len(sentence) for sentence in sequence])
    #     rankedi_word, indexs = word_lengths.sort(descending = True)
    #     ranked_indexs, inverse_indexs = indexs.sort()
    #     sequence = [sequence[i] for i in indexs]
    #     return sequence, inverse_indexs

    def forward(self, question_embeds, device,
                reverse_question_indexs, question_lengths,
                sent_alignment_model=None):
        question_packed = \
            torch.nn.utils.rnn.pack_padded_sequence(question_embeds,
                                                    question_lengths)

        question_embedding, lstm_out = self.sentence_biLstm(question_packed)
        question_embedding = question_embedding[reverse_question_indexs]

        if sent_alignment_model is not None:
            #
            reverse_question_embedding = sent_alignment_model(question_embedding)
            return reverse_question_embedding
        else:
            return question_embedding

class WordModel(nn.Module):
    """
    Model of sub-entity channel
    """
    def __init__(self, vocab_size, vocab_embedding, args):
        super(WordModel, self).__init__()
        self.batch_size = args.batch_size
        self.n_hiddens = args.n_hiddens
        self.lstm = BiLSTM(args.n_embeds, args.n_hiddens, args.batch_size)

    def attention_layer(self, question_embedding, words_embedding):
        hidden = question_embedding.view(-1, self.n_hiddens*2, 1)
        attn_weights = torch.bmm(words_embedding, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context_embeds = torch.bmm(words_embedding.transpose(1, 2),
                            soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context_embeds

    # def build_hidden(self, batch_size=1):
    #     # Before we've done anything, we dont have any hidden state.
    #     # Refer to the Pytorch documentation to see exactly
    #     # why they have this dimensionality.
    #     # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    #     return [torch.zeros(1, batch_size, self.n_hiddens),
    #             torch.zeros(1, batch_size, self.n_hiddens)]

    def init_hidden(self, device='cpu', batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.lstm.init_hidden(device, batch_size)

    # def ranking_sequence(self, sequence):
    #     word_lengths = torch.tensor([len(sentence) for sentence in sequence])
    #     rankedi_word, indexs = word_lengths.sort(descending = True)
    #     ranked_indexs, inverse_indexs = indexs.sort()
    #     sequence = [sequence[i] for i in indexs]
    #     return sequence, inverse_indexs

    def forward(self, words_embeds, device, context_embed,
                reverse_word_indexs, word_lengths,
                word_alignment_model=None):
        words_packed = \
            torch.nn.utils.rnn.pack_padded_sequence(words_embeds,
                                                    word_lengths)
        final_state, packed_out = self.lstm(words_packed)

        words_embeds_packed = \
            torch.nn.utils.rnn.pad_packed_sequence(packed_out)
        words_embeds = words_embeds_packed[0].transpose(0, 1)
        words_embeds = words_embeds[reverse_word_indexs]

        attn_output = self.attention_layer(context_embed, words_embeds)

        if word_alignment_model is not None:
            reverse_embedding = word_alignment_model(attn_output)
            return reverse_embedding
        else:
            return attn_output
