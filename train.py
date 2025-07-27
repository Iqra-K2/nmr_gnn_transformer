import os
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from dgllife.utils import RandomSplitter

import matplotlib.pyplot as plt

from dataset import GraphDataset
from util import collate_graph_spectra_to_smiles, exact_match_accuracy, token_accuracy, valid_smiles_percent
from tokenizer import load_tokenizer
from model import train_reverse_model, test_reverse_model
from gnn_models.GNNTransformerReverse import GNNTransformerReverse


def plot_training_curves(history):
    for metric in history:
        if isinstance(history[metric], dict):
            for phase in history[metric]:
                plt.plot(history[metric][phase], label=f'{metric} ({phase})')
        else:
            plt.plot(history[metric], label=metric)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.title("Training & Validation Metrics")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


def get_dataset_loaders(target, graph_representation, fold_seed, batch_size, device):
    tokenizer = load_tokenizer()

    dataset = GraphDataset(target, graph_representation)

    splitter = RandomSplitter()
    folds = splitter.k_fold_split(dataset, k=10, random_state=27407)
    trainval_set, test_set = folds[fold_seed]

    train_set, val_set = split_dataset(trainval_set, [0.95, 0.05], shuffle=True, random_state=fold_seed)

    collate = lambda batch: collate_graph_spectra_to_smiles(batch, tokenizer, device)

    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    }

    return dataset, tokenizer, loaders


def initialize_model(dataset, spectrum_len, vocab_size, hidden_dim, device):
    g = dataset[0][0]
    node_dim = g.ndata['node_attr'].shape[1]
    edge_dim = g.edata['edge_attr'].shape[1]

    model = GNNTransformerReverse(
        node_dim=node_dim,
        edge_dim=edge_dim,
        spectrum_len=spectrum_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim
    ).to(device)

    return model


def train_only(args):
    target = args.target
    graph_representation = args.graph_representation
    fold_seed = args.fold_seed

    batch_size = getattr(args, "batch_size", 32)
    spectrum_len = getattr(args, "spectrum_len", 512)
    hidden_dim = getattr(args, "hidden_dim", 256)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('./model', exist_ok=True)
    model_path = f'./model/reverse_{target}_{graph_representation}_{fold_seed}.pt'

    print("--- Loading tokenizer and dataset...")
    dataset, tokenizer, loaders = get_dataset_loaders(target, graph_representation, fold_seed, batch_size, device)
    vocab_size = tokenizer.get_vocab_size()

    print(f"--- Dataset sizes: train={len(loaders['train'].dataset)} | val={len(loaders['val'].dataset)}")

    print("--- Initializing model...")
    model = initialize_model(dataset, spectrum_len, vocab_size, hidden_dim, device)

    print("--- Training model...")
    model, history = train_reverse_model(model, loaders["train"], loaders["val"], tokenizer, model_path, device)

    print("--- Plotting training curves...")
    plot_training_curves(history)


def test_only(args):
    target = args.target
    graph_representation = args.graph_representation
    fold_seed = args.fold_seed

    batch_size = getattr(args, "batch_size", 32)
    spectrum_len = getattr(args, "spectrum_len", 512)
    hidden_dim = getattr(args, "hidden_dim", 256)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = f'./model/reverse_{target}_{graph_representation}_{fold_seed}.pt'

    print("--- Loading tokenizer and dataset...")
    dataset, tokenizer, loaders = get_dataset_loaders(target, graph_representation, fold_seed, batch_size, device)
    vocab_size = tokenizer.get_vocab_size()

    print("--- Initializing model...")
    model = initialize_model(dataset, spectrum_len, vocab_size, hidden_dim, device)

    print(f"--- Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("--- Evaluating on test set...")
    predictions, references = test_reverse_model(model, loaders["test"], tokenizer, device)

    print("\n--- Test Metrics:")
    print(f"Exact Match Accuracy: {exact_match_accuracy(predictions, references):.2f}%")
    print(f"Token-Level Accuracy: {token_accuracy(predictions, references):.2f}%")
    print(f"% Valid SMILES:       {valid_smiles_percent(predictions):.2f}%")

    print("\n--- Sample Predictions:")
    for i, (pred, ref) in enumerate(zip(predictions[:5], references[:5])):
        print(f"[{i+1}] Predicted:    {pred}")
        print(f"    Ground Truth: {ref}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="13C")
    parser.add_argument("--graph_representation", type=str, default="sparsified")
    parser.add_argument("--fold_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--spectrum_len", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")

    args = parser.parse_args()

    if args.mode == "train":
        train_only(args)
    else:
        test_only(args)
