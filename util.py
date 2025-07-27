import torch
from torch.nn.utils.rnn import pad_sequence
from dgl import batch as dgl_batch
from rdkit import Chem  # For valid SMILES checking


def collate_graph_spectra_to_smiles(batch, tokenizer, device=None):
    """
    batch: list of (graph, spectrum_tensor, smiles_string)
    Returns:
        batched_graph, node_feats, edge_feats, tgt_seq_input, tgt_seq_output, tgt_mask, tgt_key_padding_mask, node_counts
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs, spectra, smiles = zip(*batch)

    node_counts = [g.number_of_nodes() for g in graphs]

    # Batch graphs and move to device
    batched_graph = dgl_batch(graphs).to(device)
    node_feats = batched_graph.ndata['node_attr'].to(device)
    edge_feats = batched_graph.edata['edge_attr'].to(device)

    # Encode SMILES to token sequences
    token_seqs = [tokenizer.encode(s) for s in smiles]
    token_seqs = [torch.tensor(seq, dtype=torch.long, device=device) for seq in token_seqs]

    # Pad token sequences (seq_len, batch_size)
    padded_tokens = pad_sequence(token_seqs, batch_first=False, padding_value=tokenizer.pad_token_id)

    # Prepare decoder inputs and outputs by shifting tokens
    tgt_seq_input = padded_tokens[:-1, :]
    tgt_seq_output = padded_tokens[1:, :]

    # Padding mask for decoder input (True where padding)
    tgt_key_padding_mask = (tgt_seq_input == tokenizer.pad_token_id).transpose(0, 1)

    # Causal mask to prevent attending to future tokens
    seq_len = tgt_seq_input.size(0)
    tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)

    return (
        batched_graph,
        node_feats,
        edge_feats,
        tgt_seq_input,
        tgt_seq_output,
        tgt_mask,
        tgt_key_padding_mask,
        node_counts
    )


# ------------------ Helper Metrics for Test Evaluation ------------------

def exact_match_accuracy(preds, targets):
    """
    Computes the percentage of sequences that match exactly.
    """
    matches = [p == t for p, t in zip(preds, targets)]
    return sum(matches) / len(matches) * 100


def token_accuracy(preds, targets):
    """
    Computes the percentage of matching tokens across all sequences.
    """
    total, correct = 0, 0
    for p, t in zip(preds, targets):
        min_len = min(len(p), len(t))
        correct += sum([p[i] == t[i] for i in range(min_len)])
        total += len(t)
    return correct / total * 100


def valid_smiles_percent(preds):
    """
    Computes the percentage of valid SMILES using RDKit.
    """
    valid = 0
    for smile in preds:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            valid += 1
    return valid / len(preds) * 100
