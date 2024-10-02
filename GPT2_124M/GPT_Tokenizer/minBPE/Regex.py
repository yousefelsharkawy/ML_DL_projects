"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re

from base import Tokenizer, get_stats, merge


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern='GPT4'):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern == 'GPT4' else GPT2_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.inverse_special_tokens = {}


    def train(self, text, vocab_size, verbose=False):
        #print(list(map(int,"hello".encode('utf-8'))))
        assert vocab_size >= 256, "Vocab size must be at least 256"
        num_merges = vocab_size - 256

        # split the sequence using the regex pattern into text chunks
        sequence = re.findall(self.compiled_pattern, text)
        # convert each text element to bytes using utf-8 encoding (integer)
        sequence = [list(element.encode('utf-8')) for element in sequence]
        
        original_seq_length = sum(len(element) for element in sequence)
        
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):     
            # count all pairs in the sequence (for each element)
            stats = {}
            for element in sequence:
                stats = get_stats(element, counts=stats)
            
            # get the pair with most counts
            top_pair = max(stats, key=stats.get)

            # assign a new id to the new token (top_pair)
            id_ = 256 + i

            ## save the new token
            merges[top_pair] = id_ # pair -> token id
            vocab[id_] = vocab[top_pair[0]] + vocab[top_pair[1]] # token id -> byte representation (concatenation of the pair's bytes)
            
            # replace the top_pair with the new token id_ in the sequence
            sequence = [merge(element, top_pair, id_) for element in sequence]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {id_} ({vocab[id_]}) had {stats[top_pair]} occurrences")

        # overwrite the default vocab and merges
        self.vocab = vocab
        self.merges = merges

        compressed_seq_length = sum(len(element) for element in sequence)
        print("Finished training tokenizer. Compression ratio: {:.2f}".format(original_seq_length / compressed_seq_length))

    def register_special_tokens(self, special_tokens):
        """
        register special tokens to the tokenizer
        special_tokens: dict of str -> int
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    
    def encode_chunk(self,text):
        """
        encode a new text using the vocabulary we built
        this is the same as the encode function in the base class, but we will use it here for a single text chunk after regex split
        text: str
        returns: list of integers (encoded text)
        """
        # first convert the text to utf-8 encodings in integer representation
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            # use get stats to get all pairs 
            stats = get_stats(tokens)
            # get the most eligible pair 
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf"))) # get the id from the merges dict if it exists, inf if they are not in merges, then get the minimum id (should be merged first)
            # notice that in the end all of them will have inf -not mergable- and min will get one of them
            if pair not in self.merges:
                break # nothing else can be merged
            id_ = self.merges[pair]
            tokens = merge(tokens, pair, id_)
        return tokens
    
    def encode_ordinary(self,text):
        """
        encode a new text using the vocabulary we built
        """
        # split the text using the regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # encode each chunk alone
        new_tokens = []
        for chunk in text_chunks:
            new_tokens += self.encode_chunk(chunk)
        return new_tokens
    
    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens), f"special tokens {self.special_tokens} found in text"
        elif isinstance(allowed_special, set):
            # only allow the special tokens that are in the set
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            # shortcut: if we have no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        
        # we will use a regex to split the special tokens from the ordinary text
        # we use re.escape to escape the special tokens, and treat them as literals
        # we also use the | operator to combine them into a single regex pattern (ORs)
        # we use () -capturing group- 
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        
        token_sequence = []
        for element in special_chunks:
            if element in special:
                # this is a special token, so encode it separately as a special case
                token_sequence.append(special[element])
            else:
                # this is an ordinary text, so encode it normally
                token_sequence += self.encode_ordinary(element)
        
        return token_sequence
    
    def decode(self, token_sequence):
        """
        decode a list of integers into a text
        """
        text = b""
        for token in token_sequence:
            if token in self.inverse_special_tokens:
                text += self.inverse_special_tokens[token].encode("utf-8") # get the text and encode it to utf-8 bytes (everything else is utf-8 bytes)
            elif token in self.vocab:
                text += self.vocab[token] 
            else:
                raise ValueError(f"Invalid token Id {token}")
        return text.decode("utf-8")
                

        

if __name__ == "__main__":
    # test the tokenizer
    tokenizer = RegexTokenizer()
    text = """
    for i in range( 1, 101 ):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
    """
    tokenizer.train(text, 296, True)
    special_tokens = {'<|endoftext|>': len(tokenizer.vocab)}
    tokenizer.register_special_tokens(special_tokens)
    text += "<|endoftext|>"
    print(tokenizer.decode(tokenizer.encode(text,'all')) == text)
    tokenizer.save("minBPE\python_trained_models\python_regex_tokenizer")

    