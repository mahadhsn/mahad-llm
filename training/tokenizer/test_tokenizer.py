from byte_tokenizer import ByteLevelTokenizer, BOS, EOS

token = ByteLevelTokenizer(add_bos = True, add_eos = True)

def test_sample_text():
    sample_text = [
        "Hello!",
       # "My name is Mahad.",
       # "What is your name?"
    ]

    for text in sample_text:
        ids = token.encode(text)     # text -> ids
        back = token.decode(ids)     # ids -> text
        print("TEXT: ", text)
        print("IDS (first 12): ", ids[:12], "... len: ", len(ids))
        print("-" * 40)
        
def test_sample_file():
    with open("../data/raw/sample.txt", "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            ids = token.encode(line)
            back = token.decode(ids)
            print(line, "=>", ids)

test_sample_text()
# test_sample_file()
