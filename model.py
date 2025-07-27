import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GINEConv
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from rdkit import Chem
from util import exact_match_accuracy, token_accuracy


class GNNEncoder(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_dim, num_layers):
        super(GNNEncoder, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs = nn.ModuleList()
        self.convs.append(GINEConv(nn.Linear(node_feat_size, hidden_dim), self.edge_mlp))
        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(nn.Linear(hidden_dim, hidden_dim), self.edge_mlp))

    def forward(self, g, node_feats, edge_feats):
        h = node_feats

        for i, conv in enumerate(self.convs):
            h = F.relu(conv(g, h, edge_feats))

        g.ndata['h'] = h

        return h


class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, tgt_seq, memory, tgt_mask=None, tgt_key_padding_mask=None):
        embedded = self.embedding(tgt_seq) * (self.embed_size ** 0.5)

        output = self.transformer_decoder(
            embedded,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        logits = self.fc_out(output)
        return logits


class ReverseModel(nn.Module):
    def __init__(
        self,
        node_feat_size,
        edge_feat_size,
        gnn_hidden_dim,
        gnn_num_layers,
        transformer_embed_size,
        transformer_num_heads,
        transformer_num_layers,
        transformer_hidden_dim,
        vocab_size
    ):
        super(ReverseModel, self).__init__()

        self.encoder = GNNEncoder(node_feat_size, edge_feat_size, gnn_hidden_dim, gnn_num_layers)
        self.fc_memory = nn.Linear(gnn_hidden_dim, transformer_embed_size)

        self.decoder = TransformerDecoderModel(
            vocab_size=vocab_size,
            embed_size=transformer_embed_size,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            hidden_dim=transformer_hidden_dim,
        )

    def forward(self, g, node_feats, edge_feats, tgt_seq, tgt_mask=None, tgt_key_padding_mask=None, node_counts=None):
        memory = self.encoder(g, node_feats, edge_feats)

        memory = self.fc_memory(memory).unsqueeze(0)

        output = self.decoder(
            tgt_seq,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return output


def train_reverse_model(model, train_loader, val_loader, tokenizer, model_path, device, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
    best_val_loss = float('inf')

    history = {
        "loss": {"train": [], "val": []},
        "accuracy": {"train": [], "val": []},
        "perplexity": {"train": [], "val": []}
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss, total_acc, total_ppl = 0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            g, node_feats, edge_feats, tgt_seq_input, tgt_seq_output, tgt_mask, tgt_key_padding_mask, node_counts = batch
            g = g.to(device)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            tgt_seq_input = tgt_seq_input.to(device)
            tgt_seq_output = tgt_seq_output.to(device)
            if tgt_mask is not None:
                tgt_mask = tgt_mask.to(device)
            if tgt_key_padding_mask is not None:
                tgt_key_padding_mask = tgt_key_padding_mask.to(device)

            optimizer.zero_grad()

            logits = model(
                g,
                node_feats,
                edge_feats,
                tgt_seq_input,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                node_counts=node_counts
            )

            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = tgt_seq_output.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_token_id)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mask = targets_flat != pad_token_id
                correct = (logits_flat.argmax(dim=1) == targets_flat) & mask
                acc = correct.sum().item() / mask.sum().item()
                ppl = torch.exp(loss).item()

            total_loss += loss.item()
            total_acc += acc
            total_ppl += ppl

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"[train_reverse_model] Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f}, Acc: {acc:.4f}, PPL: {ppl:.2f}")

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)
        avg_train_ppl = total_ppl / len(train_loader)

        val_loss, val_acc, val_ppl = evaluate_reverse_model(model, val_loader, tokenizer, device)

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Train PPL: {avg_train_ppl:.2f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   PPL: {val_ppl:.2f}")

        history["loss"]["train"].append(avg_train_loss)
        history["loss"]["val"].append(val_loss)
        history["accuracy"]["train"].append(avg_train_acc)
        history["accuracy"]["val"].append(val_acc)
        history["perplexity"]["train"].append(avg_train_ppl)
        history["perplexity"]["val"].append(val_ppl)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

    return model, history


def evaluate_reverse_model(model, data_loader, tokenizer, device):
    model.eval()
    total_loss, total_acc, total_ppl = 0, 0, 0
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            g, node_feats, edge_feats, tgt_seq_input, tgt_seq_output, tgt_mask, tgt_key_padding_mask, node_counts = batch
            g = g.to(device)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            tgt_seq_input = tgt_seq_input.to(device)
            tgt_seq_output = tgt_seq_output.to(device)
            if tgt_mask is not None:
                tgt_mask = tgt_mask.to(device)
            if tgt_key_padding_mask is not None:
                tgt_key_padding_mask = tgt_key_padding_mask.to(device)

            logits = model(
                g,
                node_feats,
                edge_feats,
                tgt_seq_input,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                node_counts=node_counts
            )

            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = tgt_seq_output.view(-1)

            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_token_id)

            mask = targets_flat != pad_token_id
            correct = (logits_flat.argmax(dim=1) == targets_flat) & mask
            acc = correct.sum().item() / mask.sum().item()
            ppl = torch.exp(loss).item()

            total_loss += loss.item()
            total_acc += acc
            total_ppl += ppl

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_acc / len(data_loader)
    avg_ppl = total_ppl / len(data_loader)

    return avg_loss, avg_acc, avg_ppl


def test_reverse_model(model, data_loader, tokenizer, device):
    model.eval()
    pad_token_id = getattr(tokenizer, 'pad_token_id', 0)

    predictions = []
    references = []
    valid_smiles = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            g, node_feats, edge_feats, tgt_seq_input, tgt_seq_output, _, tgt_key_padding_mask, node_counts = batch
            g = g.to(device)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            tgt_seq_output = tgt_seq_output.to(device)

            if node_counts is None:
                raise ValueError("node_counts must be provided in test batches for proper padding.")

            node_embeddings = model.encoder(g, node_feats, edge_feats)

            node_embeds_list = torch.split(node_embeddings, node_counts, dim=0)

            max_nodes = max(node_counts)
            padded = []
            for node_embeds in node_embeds_list:
                pad_len = max_nodes - node_embeds.size(0)
                if pad_len > 0:
                    padding = torch.zeros(pad_len, node_embeds.size(1), device=node_embeddings.device)
                    node_embeds = torch.cat([node_embeds, padding], dim=0)
                padded.append(node_embeds)

            padded_tensor = torch.stack(padded, dim=0)

            memory = model.fc_memory(padded_tensor).permute(1, 0, 2)

            max_len, batch_size = tgt_seq_output.shape

            generated = torch.full((1, batch_size), tokenizer.start_token_id, dtype=torch.long).to(device)

            for step in range(max_len - 1):
                logits = model.decode(generated, memory)

                next_token_logits = logits[-1, :, :]

                next_tokens = next_token_logits.argmax(dim=-1, keepdim=True).transpose(0, 1)
                generated = torch.cat([generated, next_tokens], dim=0)

            generated = generated.transpose(0, 1).cpu().tolist()
            tgt_seq_output = tgt_seq_output.transpose(0, 1).cpu().tolist()

            def decode_trim(ids):
                if tokenizer.pad_token_id in ids:
                    ids = ids[:ids.index(tokenizer.pad_token_id)]
                eos_id = tokenizer.tokenizer.token_to_id("<end>")
                if eos_id in ids:
                    ids = ids[:ids.index(eos_id)]
                return tokenizer.decode(ids)

            batch_valid_smiles = 0
            for pred_ids, tgt_ids in zip(generated, tgt_seq_output):
                pred_smiles = decode_trim(pred_ids)
                tgt_smiles = decode_trim(tgt_ids)

                predictions.append(pred_smiles)
                references.append(tgt_smiles)

                if Chem.MolFromSmiles(pred_smiles) is not None:
                    valid_smiles += 1
                    batch_valid_smiles += 1

            total_samples += batch_size

    exact = exact_match_accuracy(predictions, references)
    token = token_accuracy(predictions, references)
    valid = 100 * valid_smiles / len(predictions) if predictions else 0.0

    print("\n=== Final Test Metrics ===")
    print(f"Exact Match Accuracy: {exact:.2f}%")
    print(f"Token-level Accuracy: {token:.2f}%")
    print(f"Valid SMILES: {valid:.2f}%")

    return predictions, references

