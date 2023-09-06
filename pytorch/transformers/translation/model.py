import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(Embedding, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model), we create matrix of this size as the longest sentence can be of seq_len
        pe = torch.zeros(seq_len, d_model)  # positional_encoding

        # pe(pos, 2i) = sin(pos/(10000^(2i/d_model)))
        # pe(pos, 2i + 1) = cos(pos/(10000^(2i/d_model)))
        # we will do a slightly modified calculation using log-space, as it is more numerically stable; this is based on the fact that when we do exponential and take log of the number, the resultant is the same number but is more stable

        # (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # apply sin to even postions and cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # adding a batch dimension to apply to batches of sentences; (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe", pe
        )  # register a buffer that should not be considered a model parameter (learned parameter), but is part of the module's state

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)

        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(
        self, eps: float = 1e-6
    ):  # eps is introduced for numerial stability and to avoid division by 0
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))  # multiplicative
        self.beta = nn.Parameter(torch.zeros(1))  # additive

    def forward(self, x):
        mean = x.mean(
            dim=-1, keepdim=True
        )  # do the mean to everything after the batch; keepdim is necessary as usually mean cancels the dimension on which the mean is applied
        std = x.std(dim=-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForwardBlock, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),  # W1 and B1
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),  # W2 and B2
        )

    def forward(self, x):
        return self.ff(
            x
        )  # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by heads"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(
            d_k
        )  # torch.matmul and @ are equivalent only for a rank 2 tensor. The @ operaion is "actually" torch.bmm (batch matrix multiply) in which the matrix multiply is done on the last two dimensions

        # hide the interation between words
        if mask is not None:
            # attention_scores.masked_fill(mask == 0, -1e9)
            attention_scores.masked_fill_(
                mask == 0, -1e30 if attention_scores.dtype == torch.float32 else -1e4
            )  # mixed-precision operations

        # # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (
            attention_scores @ value
        ), attention_scores  # the second is used for visualization

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # splitting to give to multiple heads; (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        # we transpose as we prefer the h dim to be the second dimension
        split_and_transpose = lambda tensor: tensor.view(
            tensor.shape[0], tensor.shape[1], self.h, self.d_k
        ).transpose(1, 2)

        query = split_and_transpose(query)
        key = split_and_transpose(key)
        value = split_and_transpose(value)

        x, self.atten_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # (batch, h, seq_len, d_k) -> # (batch, seq_len, h, d_k) -> # (batch, seq_len, d_model)
        # https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
        # pytorch doesn't generate a new tensor with a new layout, it just modifies meta information in the Tensor object so that the offset and stride describe the desired new shape.
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super(ResidualConnection, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # add and norm
        return x + self.dropout(
            sublayer(self.norm(x))
        )  # in the paper, they apply first the sublayer then the normalization


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super(EncoderBlock, self).__init__()

        self.self_attention_block = self_attention_block  # it is called the self_attention because we give the query, keys and values is x itself, that is the words' attention to itself and others in the same sentence
        self.feed_forward_block = feed_forward_block
        self.dropout = dropout
        self.residual_connetions = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(
        self, x, src_mask
    ):  # we take the src_mask to hide the interaction of the padding with other words
        x = self.residual_connetions[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connetions[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):  #! probably will be good to just take Nx
        super(Encoder, self).__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)  #! this is already applied, right?


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super(DecoderBlock, self).__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    # src_mask is the mask applied to the encoder, and tgt_mask is the mask applied to the decoder
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):  #! probably will be good to just take Nx
        super(Decoder, self).__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)  #! this is already applied, right?


# we need to map the byte pair encoding to the vocabulary
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(ProjectionLayer, self).__init__()

        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(
            self.proj(x), dim=-1
        )  # log softmax for numerical stability


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: Embedding,
        tgt_embed: Embedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    Nx: int = 6,  # number of encoder and decoder blocks
    h: int = 8,  # number of heads
    dropout: float = 0.1,
    d_ff: int = 2048,  # hidden dimension for the feed forward network
) -> Transformer:
    # create the embedding layers
    src_embed = Embedding(d_model, src_vocab_size)
    tgt_embed = Embedding(d_model, tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(Nx):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(Nx):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
