import torch
import abc

class ContextEncoder(torch.nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()

    @abc.abstractmethod
    def forward(
        self, 
        word_list, 
        word_lengths,
        is_encoder_side=None,
    ):
        pass

    def embed_contexts(
        self, 
        word_list,
        word_lengths,
        num_words_per_utt,
        is_encoder_side=None,
    ):
        """
        Args:
            word_list: 
                Option1: A list of words, where each word is a list of token ids.
                The list of tokens for each word has been padded.
                Option2: A list of words, where each word is an embedding.
            word_lengths:
                Option1: The number of tokens per word
                Option2: None
            num_words_per_utt:
                The number of words in the context for each utterance
        Returns:
            final_h:
                A tensor of shape (batch_size, max(num_words_per_utt) + 1, joiner_dim),
                which is the embedding for each context word.
            mask_h:
                A tensor of shape (batch_size, max(num_words_per_utt) + 1),
                which contains a True/False mask for final_h
        """

        # print(f"word_list.shape={word_list.shape}")
        final_h = self.forward(word_list, word_lengths, is_encoder_side=is_encoder_side)

        final_h = torch.split(final_h, num_words_per_utt)
        final_h = torch.nn.utils.rnn.pad_sequence(
            final_h, 
            batch_first=True, 
            padding_value=0.0
        )
        # print(f"final_h.shape={final_h.shape}")

        # add one no-bias token
        no_bias_h = torch.zeros(final_h.shape[0], 1, final_h.shape[-1])
        no_bias_h = no_bias_h.to(final_h.device)
        final_h = torch.cat((no_bias_h, final_h), dim=1)
        # print(final_h)

        # https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
        mask_h = torch.arange(max(num_words_per_utt) + 1)
        mask_h = mask_h.expand(len(num_words_per_utt), max(num_words_per_utt) + 1) > torch.Tensor(num_words_per_utt).unsqueeze(1)
        mask_h = mask_h.to(final_h.device)

        return final_h, mask_h

    def clustering(self):
        pass

    def cache(self):
        pass
