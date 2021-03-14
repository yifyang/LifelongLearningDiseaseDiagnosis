import torch
import torch.nn as nn

from model_utils import SentenceModel, WordModel

class MultiModel(nn.Module):
    """
    Combined model of sub-entity model and context model
    """
    def __init__(self, vocab_size, vocab_embedding, args):
        super(MultiModel, self).__init__()
        self.batch_size = args.batch_size
        self.device = args.device
        self.word_embeddings = nn.Embedding(vocab_size, args.n_embeds)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(vocab_embedding))
        self.word_embeddings = self.word_embeddings.to(args.device)
        self.word_embeddings.weight.requires_grad = False
        self.wordModel = WordModel(vocab_size, vocab_embedding, args)
        self.sentenceModel = SentenceModel(vocab_size, vocab_embedding, args)
        self.output = nn.Linear(args.n_hiddens * 4, args.n_outputs)

    def compute_sent_embed(self, sentence_list, sentence_lengths,
                          reverse_sentence_indexs, alignemnt_model,
                          before_alignment=False):
        """
        Used to compute the embedding of sentence_list

        :param sentence_list: samples
        :param sentence_lengths: list of lengths of all samples
        :param reverse_sentence_indexs: indexes of samples before ranking
        :param alignemnt_model: sentence alignment model
        :param before_alignment: embedding before or after alignment
        :return: embedding of all samples
        """
        sentence_embeds = self.word_embeddings(sentence_list)
        self.sentenceModel.init_hidden(self.device, len(sentence_lengths))

        if alignemnt_model is not None and not before_alignment:
            sentence_embedding = self.sentenceModel(sentence_embeds, self.device,
                                                    reverse_sentence_indexs,
                                                    sentence_lengths,
                                                    alignemnt_model)
        else:
            sentence_embedding = self.sentenceModel(sentence_embeds, self.device,
                                                    reverse_sentence_indexs,
                                                    sentence_lengths,
                                                    sent_alignment_model=None)
        return sentence_embedding.detach()

    def compute_word_embed(self, word_list, word_lengths, context_embeds,
                           reverse_word_indexs, alignemnt_model,
                           before_alignment=False):
        """
        Used to compute the embedding of entity_list (word_list)

        :param word_list: samples
        :param word_lengths: list of lengths of all samples
        :param context_embeds: embedding of corresponding sentences
        :param reverse_word_indexs: indexes of samples before ranking
        :param alignemnt_model: word alignment model
        :param before_alignment: embedding before or after alignment
        :return: embedding of samples
        """
        word_embeds = self.word_embeddings(word_list)
        self.wordModel.init_hidden(self.device, len(word_lengths))

        if alignemnt_model is not None and not before_alignment:
            sum_word_embeds = self.wordModel(word_embeds, self.device,
                                             context_embeds,
                                             reverse_word_indexs,
                                             word_lengths,
                                             alignemnt_model)
        else:
            sum_word_embeds = self.wordModel(word_embeds, self.device,
                                             context_embeds,
                                             reverse_word_indexs,
                                             word_lengths,
                                             word_alignment_model=None)

        return sum_word_embeds.detach()

    def forward(self, sentence_list, word_list,
                reverse_sentence_indexs, reverse_word_indexs,
                sentence_lengths, word_lengths,
                word_alignment_model=None, sent_alignment_model=None):
        sentence_embeds = self.word_embeddings(sentence_list)

        self.sentenceModel.init_hidden(self.device, len(sentence_lengths))
        sentence_embeds = self.sentenceModel(sentence_embeds, self.device,
                                             reverse_sentence_indexs,
                                             sentence_lengths,
                                             sent_alignment_model)

        word_embeds = self.word_embeddings(word_list)
        self.wordModel.init_hidden(self.device, len(word_lengths))
        sum_word_embeds = self.wordModel(word_embeds, self.device,
                                         sentence_embeds,
                                         reverse_word_indexs,
                                         word_lengths,
                                         word_alignment_model)

        concat_embedding = torch.cat((sum_word_embeds, sentence_embeds), dim=-1)
        output = self.output(concat_embedding)

        return output


