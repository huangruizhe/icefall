import torch
import torch.nn as nn


class BiasingModule(torch.nn.Module):
    def __init__(
        self, 
        query_dim,
        qkv_dim=64,
        num_heads=4,
    ):
        super(BiasingModule, self).__init__()
        self.proj_in = nn.Linear(query_dim, qkv_dim)
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=qkv_dim,
            num_heads=num_heads, 
            # kdim=64,
            # vdim=64,
            batch_first=True
        )
        self.proj_out = nn.Linear(qkv_dim, query_dim)

    def forward(
        self, 
        queries,
        contexts,
        contexts_mask,
        need_weights=False,
    ):
        """
        Args:
            query: 
                of shape batch_size * seq_length * query_dim
            contexts: 
                of shape batch_size * max_contexts_size * query_dim
            contexts_mask:
                of shape batch_size * max_contexts_size
        Returns:
            attn_output:
                of shape batch_size * seq_length * context_dim
        """

        queries = self.proj_in(queries)
        
        attn_output, attn_output_weights = self.multihead_attn(
            queries,    # query
            contexts,   # key
            contexts,   # value
            key_padding_mask=contexts_mask,
            need_weights=need_weights,
        )
        output = self.proj_out(attn_output)

        # print(f"query={query.shape}")
        # print(f"value={contexts} value.shape={contexts.shape}")
        # print(f"attn_output_weights={attn_output_weights} attn_output_weights.shape={attn_output_weights.shape}")
        return output, attn_output_weights
