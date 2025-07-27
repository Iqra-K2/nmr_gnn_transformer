import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import dgl
from dgl.nn.pytorch import GINEConv


class GNNEncoderWithEdges(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_dim, num_layers):
        super().__init__()
        self.node_feat_proj = nn.Linear(node_feat_size, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
            for _ in range(num_layers)
        ])

    def forward(self, g, node_feats, edge_feats):
        h = F.relu(self.node_feat_proj(node_feats))
        e = self.edge_mlp(edge_feats)

        for i, conv in enumerate(self.convs):
            h = conv(g, h, e)
            h = F.relu(h)

        return h 


class GNNTransformerReverse(nn.Module):
    def __init__(self, node_dim, edge_dim, spectrum_len, vocab_size, hidden_dim,
                 gnn_num_layers=3, transformer_embed_size=256,
                 transformer_num_heads=8, transformer_num_layers=3,
                 transformer_hidden_dim=512):
        super().__init__()
        self.encoder = GNNEncoderWithEdges(node_dim, edge_dim, hidden_dim, gnn_num_layers)
        self.fc_memory = nn.Linear(hidden_dim, transformer_embed_size)

        decoder_layer = TransformerDecoderLayer(
            d_model=transformer_embed_size,
            nhead=transformer_num_heads,
            dim_feedforward=transformer_hidden_dim
        )
        self.decoder = TransformerDecoder(decoder_layer, transformer_num_layers)

        self.embedding = nn.Embedding(vocab_size, transformer_embed_size)
        self.fc_out = nn.Linear(transformer_embed_size, vocab_size)

    def forward(self, g, node_feats, edge_feats, tgt_seq, tgt_mask=None, tgt_key_padding_mask=None, node_counts=None):
        h = self.encoder(g, node_feats, edge_feats)

        node_embeds_list = torch.split(h, node_counts, dim=0)

        max_nodes = max(node_counts)
        padded = []
        for idx, node_embeds in enumerate(node_embeds_list):
            pad_len = max_nodes - node_embeds.size(0)
            if pad_len > 0:
                padding = torch.zeros(pad_len, node_embeds.size(1), device=h.device)
                node_embeds = torch.cat([node_embeds, padding], dim=0)
            padded.append(node_embeds)

        padded_tensor = torch.stack(padded, dim=0)

        memory = self.fc_memory(padded_tensor)
        memory = memory.permute(1, 0, 2)

        embedded = self.embedding(tgt_seq)

        output = self.decoder(
            embedded,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = self.fc_out(output)

        return output

    def decode(self, tgt_seq, memory, tgt_mask=None, tgt_key_padding_mask=None):
        embedded = self.embedding(tgt_seq)
        output = self.decoder(embedded, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.fc_out(output)
        return output