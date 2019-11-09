import tokenization

tokenizer = tokenization.FullTokenizer(
        vocab_file="chinese_L-12_H-768_A-12/vocab.txt", do_lower_case=False)

print(tokenizer.convert_ids_to_tokens(list(123)))