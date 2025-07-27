
from tokenizers import Tokenizer

class SmilesTokenizer:
    def __init__(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.start_token = "<start>"
        self.end_token = "<end>"

        # Get token IDs for special tokens
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)
        self.unk_token_id = self.tokenizer.token_to_id(self.unk_token)
        self.start_token_id = self.tokenizer.token_to_id(self.start_token)
        self.end_token_id = self.tokenizer.token_to_id(self.end_token)

    def encode(self, smiles):
        ids = self.tokenizer.encode(smiles).ids
        return [self.start_token_id] + ids + [self.end_token_id]

    def decode(self, ids, skip_special_tokens=True):
        # Optionally skip <start> and <end> tokens
        if skip_special_tokens:
            ids = [id for id in ids if id not in {self.start_token_id, self.end_token_id, self.pad_token_id}]
        return self.tokenizer.decode(ids)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()


def load_tokenizer(path="tokenizers/tokenizer_vocab_100.json"):
    return SmilesTokenizer(path)