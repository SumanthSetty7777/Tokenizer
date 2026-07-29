from src.bpe import BPETokenizer, train_bpe_fast_v1, train_bpe_slow

text = "Hello, how are you?<|endoftext|>Héllò hôw are ü? 🙃"
# spec = train_bpe_fast_v1(text, vocab_size=320, special_tokens=("<|endoftext|>",))
# tokenizer = BPETokenizer.from_spec(spec)

# ids = tokenizer.encode(text)
# print(tokenizer.decode(ids) == text)


slow = train_bpe_slow(text, 300, ("<|endoftext|>",))
fast = train_bpe_fast_v1(text, 300, ("<|endoftext|>",))

print(slow.merges == fast.merges)
print(slow.vocab == fast.vocab)