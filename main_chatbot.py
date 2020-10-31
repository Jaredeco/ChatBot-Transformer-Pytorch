print("Starting ChatBot...")
import torch
from model import Transformer
import spacy


def read_vocab(path):
    vocab = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token[:-1]] = int(index)
    return vocab


spacy_en = spacy.load("en_core_web_sm")
net = torch.load("chatbot_model.pth").cuda()
net.eval()
src_vocab = read_vocab("src_vocab.txt")
trg_vocab = read_vocab("trg_vocab.txt")
test_sentence = "Who suggested Lincoln grow a beard?"


def predict(test_sentence, model):
    max_len = 100
    sen = [w.text.lower() for w in spacy_en.tokenizer(test_sentence)]
    sen.insert(0, src_vocab["<sos>"])
    sen.append(src_vocab["<eos>"])
    inp_sen = [(src_vocab[i] if i in src_vocab else src_vocab["<unk>"]) for i in sen]
    inp_sen = torch.tensor(inp_sen, dtype=torch.long).unsqueeze(1).cuda()
    outputs = [trg_vocab["<sos>"]]
    for i in range(max_len):
        trg = torch.tensor(outputs, dtype=torch.long).unsqueeze(1).cuda()
        with torch.no_grad():
            output = model(inp_sen, trg)
        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)
        if best_guess == trg_vocab["<eos>"]:
            break
    itos = dict((v, k) for k, v in trg_vocab.items())
    pred_sentence = [(itos[i] if i in itos else "<unk>") for i in outputs]
    return pred_sentence


while True:
    inp_sen = input("Ask me something: ")
    pred = predict(inp_sen, net)
    print(" ".join(pred[1:-1]))
