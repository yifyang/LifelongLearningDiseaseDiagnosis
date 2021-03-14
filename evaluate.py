import torch
from utils import ranking_sequence

def process_testing_samples(sample_list, device):
    sentences = []
    targets = []
    all_words = []
    for sample in sample_list:
        sentence = torch.tensor(sample[1], dtype=torch.long).to(device)
        words = torch.tensor(sample[2], dtype=torch.long).to(device)
        targets.append(sample[0]-1)

        all_words.append(words)
        sentences += [sentence]

    targets = torch.tensor(targets, dtype=torch.long).to(device)
    return sentences, targets, all_words


def evaluate_model(model, testing_data, batch_size, device,
                   word_alignment_model=None, sent_alignment_model=None):
    num_correct = 0
    for i in range((len(testing_data)-1)//batch_size+1):
        samples = testing_data[i*batch_size:(i+1)*batch_size]
        sentences, targets, words = \
            process_testing_samples(samples, device)

        ranked_words, alignment_words_indexs = \
            ranking_sequence(words)
        ranked_sentences, alignment_sentence_indexs = \
            ranking_sequence(sentences)

        words_lengths = [len(word) for word in ranked_words]
        sentence_lengths = [len(sentence) for sentence in ranked_sentences]

        pad_sentences = torch.nn.utils.rnn.pad_sequence(ranked_sentences)
        pad_sentences = pad_sentences.to(device)
        pad_words = torch.nn.utils.rnn.pad_sequence(ranked_words)
        pad_words = pad_words.to(device)

        output = model(pad_sentences, pad_words,
                   alignment_sentence_indexs, alignment_words_indexs,
                   sentence_lengths, words_lengths,
                   word_alignment_model, sent_alignment_model)

        for j, one_pred in enumerate(output):
            if torch.argmax(one_pred) == targets[j]:
                num_correct += 1

    return float(num_correct)/len(testing_data)