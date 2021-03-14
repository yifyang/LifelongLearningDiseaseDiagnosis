import torch
import numpy as np

def ranking_sequence(sequence):
    """
    rank samples with length
    """
    word_lengths = torch.tensor([len(sentence) for sentence in sequence])
    rankedi_word, indexs = word_lengths.sort(descending = True)
    ranked_indexs, inverse_indexs = indexs.sort()
    sequence = [sequence[i] for i in indexs]
    return sequence, inverse_indexs

def get_word_embed(model, sample_list, batch_size, device, alignment_model,
                   before_alignment=False):
    """
    Retrieve the embedding of sub entities

    :param model: The whole model
    :param sample_list: memory data of previous tasks
    :param device: GPU or CPU
    :param alignment_model: Alignment model for entities
    :param before_alignment: Embeddings before or after the alignment
    :return: embeddings of sample_list
    """
    ret_word_embeds = []
    for i in range((len(sample_list) - 1) // batch_size + 1):
        samples = sample_list[i * batch_size:(i + 1) * batch_size]
        batch_words = []
        batch_sentences = []
        for item in samples:
            this_words = torch.tensor(item[2], dtype=torch.long).to(device)
            batch_words.append(this_words)

            # corresponding sentences are used for attention part
            this_sentence = torch.tensor(item[1], dtype=torch.long).to(device)
            batch_sentences.append(this_sentence)
        ranked_words, alignment_word_indexs = \
            ranking_sequence(batch_words)
        word_lengths = [len(words) for words in ranked_words]

        ranked_sentences, alignment_sentence_indexs = \
            ranking_sequence(batch_sentences)
        sentence_lengths = [len(sentence) for sentence in ranked_sentences]
        pad_sentences = torch.nn.utils.rnn.pad_sequence(ranked_sentences)
        que_embeds = model.compute_sent_embed(pad_sentences,
                                             sentence_lengths,
                                             alignment_sentence_indexs,
                                             alignment_model, before_alignment)

        pad_words = torch.nn.utils.rnn.pad_sequence(ranked_words)
        pad_words = pad_words.to(device)
        word_embeds = model.compute_word_embed(pad_words, word_lengths,
                                               que_embeds,
                                               alignment_word_indexs,
                                               alignment_model, before_alignment)
        ret_word_embeds.append(word_embeds.detach().cpu().numpy())

    return np.concatenate(ret_word_embeds)

def get_sent_embed(model, sample_list, batch_size, device, alignment_model,
                  before_alignment=False):
    """
    Retrieve the embedding of contexts

    :param model: The whole model
    :param sample_list: memory data of previous tasks
    :param device: GPU or CPU
    :param alignment_model: Alignment model for contexts
    :param before_alignment: Embeddings before or after the alignment
    :return: embeddings of sample_list
    """
    ret_que_embeds = []
    for i in range((len(sample_list)-1)//batch_size+1):
        samples = sample_list[i*batch_size:(i+1)*batch_size]
        batch_sentences = []
        for item in samples:
            this_sentence = torch.tensor(item[1], dtype=torch.long).to(device)
            batch_sentences.append(this_sentence)

        ranked_sentences, alignment_sentence_indexs = \
            ranking_sequence(batch_sentences)
        sentence_lengths = [len(sentence) for sentence in ranked_sentences]
        pad_sentences = torch.nn.utils.rnn.pad_sequence(ranked_sentences)
        que_embeds = model.compute_sent_embed(pad_sentences,
                                             sentence_lengths,
                                             alignment_sentence_indexs,
                                             alignment_model, before_alignment)
        ret_que_embeds.append(que_embeds.detach().cpu().numpy())
    return np.concatenate(ret_que_embeds)