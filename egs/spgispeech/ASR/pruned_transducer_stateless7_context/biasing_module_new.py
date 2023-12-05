import torch
import torch.nn as nn


class Ffn(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, nlayers=1, drop_out=0.1, skip=False) -> None:
        super().__init__()

        layers = []
        for ilayer in range(nlayers):
            _in = hidden_dim if ilayer > 0 else input_dim
            _out = hidden_dim if ilayer < nlayers - 1 else out_dim
            layers.extend([
                nn.Linear(_in, _out),
                # nn.ReLU(),
                nn.Sigmoid(),
                nn.Dropout(p=drop_out),
            ])
        self.ffn = torch.nn.Sequential(
            *layers,
        )

        self.skip = skip
    
    def forward(self, x) -> torch.Tensor:
        x_out = self.ffn(x)
        
        if self.skip:
            x_out = x_out + x

        return x_out
    

class BiasingModule(torch.nn.Module):
    def __init__(
        self, 
        query_dim,
        qkv_dim=64,
        num_heads=4,
    ):
        super(BiasingModule, self).__init__()
        self.proj_in1 = Ffn(
            input_dim=query_dim,
            hidden_dim=query_dim,
            out_dim=query_dim,
            skip=False,
            drop_out=0.2,
            nlayers=3,
        )
        self.proj_in2 = nn.Linear(query_dim, qkv_dim)
        self.multihead_attn = torch.nn.MultiheadAttention(
            embed_dim=qkv_dim,
            num_heads=num_heads, 
            # kdim=64,
            # vdim=64,
            batch_first=True,
        )
        self.proj_out1 = nn.Linear(qkv_dim, query_dim)
        self.proj_out2 = Ffn(
            input_dim=query_dim,
            hidden_dim=query_dim,
            out_dim=query_dim,
            skip=False,
            drop_out=0.2,
            nlayers=3,
        )
        self.glu = nn.GLU()

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

        # queries = self.proj_in1(queries)
        queries = self.proj_in2(queries)
        # queries = queries / 0.01
        
        attn_output, attn_output_weights = self.multihead_attn(
            queries,    # query
            contexts,   # key
            contexts,   # value
            key_padding_mask=contexts_mask,
            need_weights=need_weights,
        )
        output = self.proj_out1(attn_output)
        # output = self.proj_out2(output)

        # apply the gated linear unit
        # output = self.glu(output.repeat(1,1,2))

        # print(f"query={query.shape}")
        # print(f"value={contexts} value.shape={contexts.shape}")
        # print(f"attn_output_weights={attn_output_weights} attn_output_weights.shape={attn_output_weights.shape}")
        return output, attn_output_weights
