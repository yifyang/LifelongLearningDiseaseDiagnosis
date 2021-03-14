import numpy as np

def read_file(path):
    with open(path, "r", encoding="utf8") as f:
        init_data = f.readlines()

    pro_data = [[int(line.split("\t")[0]), eval(line.split("\t")[2]), eval(line.split("\t")[3])]
                for line in init_data]

    return pro_data

def read_fasttext(fasttext_file):
    fasttext_vocabulary = []
    fasttext_embedding = {}
    with open(fasttext_file, "r", encoding="utf8") as file_in:
        for line in file_in:
            items = line.split()
            word = items[0]
            fasttext_vocabulary.append(word)
            fasttext_embedding[word] = np.asarray(items[1:], dtype='float32')
    return fasttext_vocabulary, fasttext_embedding

def split_data(dataset, num_clusters, cluster_labels):
    """
    split the data into sequential tasks
    """
    splited_data = [[] for _ in range(num_clusters)]
    for data in dataset:
        cluster_number = cluster_labels[data[0]]
        splited_data[cluster_number].append(data)

    return splited_data

def build_vocabulary_embedding(all_samples, embedding_size=300, fasttext_path=None):
    fasttext_vocabulary, fasttext_embedding = read_fasttext(fasttext_path)

    vocabulary = {}
    embedding = []
    index = 0
    np.random.seed(100)
    for sample in all_samples:
        all_words = sample[1] + sample[2]
        for word in all_words:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
                # init the word that are not in fasttext vocabulary randomly
                if word in fasttext_embedding:
                    embedding.append(fasttext_embedding[word])
                else:
                    embedding.append(np.random.rand(embedding_size))

    return vocabulary, embedding

def word2index(dataset, vocab):
    for data in dataset:
        temp_sent = [vocab[word] for word in data[1]]
        data[1] = temp_sent
        temp_word = [vocab[word] for word in data[2]]
        data[2] = temp_word

    return dataset

def gen_data(train_path, test_path, args, fasttext_path=None):
    """
    preprocess all the data
    """
    train_word_data = read_file(train_path)
    test_word_data = read_file(test_path)

    all_samples = train_word_data + test_word_data
    vocab, embed = build_vocabulary_embedding(all_samples, args.n_embeds, fasttext_path)

    train_index_data = word2index(train_word_data, vocab)
    test_index_data = word2index(test_word_data, vocab)

    return train_index_data, test_index_data, vocab, embed