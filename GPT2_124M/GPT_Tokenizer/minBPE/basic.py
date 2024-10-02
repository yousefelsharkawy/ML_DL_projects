"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self,text,vocab_size,verbose=False):
        assert vocab_size >= 256
        num_merges =  vocab_size - 256
        sequence = text.encode('utf-8') # utf-8 bytes (hexadecimal representation)
        sequence = list(map(int,sequence)) # utf-8 bytes (integer representation)
        original_length = len(sequence)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for i in range(num_merges):
            # count all pair combinations
            stats = get_stats(sequence)
            # get the most common pair
            top_pair = max(stats, key=stats.get)
            # assign a new id to the new token (top pair)
            id_ = 256 + i
            
            ## save the new token
            # we will use merges to encode (we need it to know the order of merges to make)
            merges[top_pair] = id_
            # we will use vocab to decode (we can construct it from merges but this way will decode faster)
            vocab[id_] = vocab[top_pair[0]] + vocab[top_pair[1]]
            # replace that top pair with the new token and repeat the process
            sequence = merge(sequence,top_pair,id_)
            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {id_} ({vocab[id_]}) had {stats[top_pair]} occurrences")

        # override the class variables
        self.merges = merges
        self.vocab = vocab
            
        print(f"Finished Training, Compression ratio: {original_length/len(sequence):.2f}X")

    def encode(self,text):
        """
        encode a new text using the vocabulary we built
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
    
    def decode(self,tokens):
        utf_8_sequence = b"".join([self.vocab[tok] for tok in tokens])
        return utf_8_sequence.decode('utf-8',errors='replace')
        
        




if __name__ == "__main__":
    # test the tokenizer
    tokenizer = BasicTokenizer()
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
    print(tokenizer.decode(tokenizer.encode(text)) == text)
    