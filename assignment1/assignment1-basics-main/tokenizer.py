import regex as re

class Tokenizer:

    def __init__(self, vocab_size = 0, special_tokens = [], regex_pattern = None, merges = {}, vocab = {}, merge_list = []):
        self.special_tokens = special_tokens
        self.regex_pattern = regex_pattern
        self.vocab_size = vocab_size
        self.merges = merges
        self.vocab = vocab
        self.merge_list = merge_list

    def stats(self, tokens, count = None):
        count = count if count is not None else {}
        
        for pair in zip(tokens, tokens[1:]):
            count[pair] = 1 + count.get(pair,0)
        return count
    
    def merge(self, tokens, pair, new_idx):
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(new_idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def encode(self, text):
        pass

    def decode(self, tokens):
        pass

    def apply_regex(self, text):
        if not self.regex_pattern:
            return [[text]]

        # Step 1: Special tokens ko markers se separate karo
        # "Hello <pad> world" → ["Hello ", "<pad>", " world"]
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            parts = re.split(special_pattern, text)
        else:
            parts = [text]

        # Step 2: Har part ko handle karo
        tokens = []
        pattern = re.compile(self.regex_pattern)

        for part in parts:
            if part in self.special_tokens:
                # Special token → preserve as-is (encoding ke liye)
                continue
                tokens.append(part)
            elif part:
                # Regular text → regex se chunks nikalo
                chunks = pattern.findall(part)
                for chunk in chunks:
                    tokens.append(list(chunk.encode('utf-8')))

        return tokens

    def training(self, text):
        tokens = self.apply_regex(text)
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        for num_merges in range(self.vocab_size - 256 - len(self.special_tokens)):
            counts = {}
            for chunk in tokens:
                self.stats(chunk, counts)

            if not counts:
                break

            max_pair = max(counts, key = lambda item: (counts[item], self.vocab[item[0]], self.vocab[item[1]]))
            new_idx = 256 + num_merges
            self.merges[max_pair] = new_idx
            self.vocab[new_idx] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
            self.merge_list.append((self.vocab[max_pair[0]], self.vocab[max_pair[1]]))

            for chunk in range(len(tokens)):
                tokens[chunk] = self.merge(tokens[chunk], max_pair, new_idx)

        self.register_special_tokens()

    def register_special_tokens(self):
        for idx, token in enumerate(self.special_tokens):
            new_idx = len(self.vocab) + idx
            self.vocab[new_idx] = token.encode("utf-8")