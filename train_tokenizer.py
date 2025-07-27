
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizers.normalizers import Sequence, NFKC
from tokenizers.processors import ByteLevel
import os
import argparse
import numpy as np


def load_smiles_from_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "data" not in data:
        raise ValueError("The .npz file does not contain a 'data' key.")

    smiles_array = data["data"].item()["smi"]
    return smiles_array.tolist()


def build_tokenizer_from_smiles(smiles_list, save_path, pad_len=100, vocab_size=100):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = ByteLevel(trim_offsets=True)

    tokenizer.enable_truncation(max_length=pad_len)
    tokenizer.enable_padding(length=pad_len, pad_id=0, pad_token="<pad>")

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<start>", "<end>"]
    )

    tokenizer.train_from_iterator(smiles_list, trainer=trainer)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f" Tokenizer saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to the processed .npz file containing SMILES")
    parser.add_argument("--output", type=str, default="tokenizers/tokenizer_vocab_100.json")
    parser.add_argument("--pad_len", type=int, default=100)
    parser.add_argument("--vocab_size", type=int, default=100)
    args = parser.parse_args()

    smiles_list = load_smiles_from_npz(args.npz)
    build_tokenizer_from_smiles(smiles_list, args.output, args.pad_len, args.vocab_size)
