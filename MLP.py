import torch
import torch.nn.functional as F
import random
import time
start_time = time.time()
words = open("names.txt",
             "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
stoi["."] = 0
itos[0] = "."
block_size = 4


def dataset(words):
    X, Y = [], []
    for w in words:
        # print(w)
        context = [0]*block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print("".join(itos[i] for i in context), '----->', itos[ix])
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    # print(X.shape, Y.shape)
    return X, Y


random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

X_train, Y_train = dataset(words[:n1])
X_val, Y_val = dataset(words[n1:n2])
X_test, Y_test = dataset(words[n2:])
C = torch.randn((27, 15))
W1 = torch.randn((60, 300))
B1 = torch.randn(300)
W2 = torch.randn((300, 27))
B2 = torch.randn(27)
params = [C, W1, B1, W2, B2]

for p in params:
    p.requires_grad = True

sum(p.nelement() for p in params)
lri = []
lossi = []
stepi = []

for _ in range(200000):
    # minibatch
    ix = torch.randint(0, X_train.shape[0], (64,))
    # forward pass
    emb = C[X_train[ix]]
    h = torch.tanh(emb.view(-1, 60) @ W1 + B1)
    logits = h @ W2 + B2
    loss = F.cross_entropy(logits, Y_train[ix])
    # print(loss.item())
    # backward pass
    for p in params:
        p.grad = None
    loss.backward()
    # update
    # lr = lrs[_]
    lr = 0.1 if _ < 100000 else 0.01
    for p in params:
        p.data += -lr * p.grad
    # track
    # lri.append(lre[_])
    lossi.append(loss.log10().item())
    stepi.append(_)

print("Validation loss:")
emb = C[X_val]
h = torch.tanh(emb.view(-1, 60) @ W1 + B1)
logits = h @ W2 + B2
# counts = logits.exp()
# prob = counts/counts.sum(1, keepdims= True)
# loss = -prob[torch.arange(32), Y].log().mean()
val_loss = F.cross_entropy(logits, Y_val)
print(val_loss.item())
print("\nTest loss:")
emb = C[X_test]
h = torch.tanh(emb.view(-1, 60) @ W1 + B1)
logits = h @ W2 + B2
test_loss = F.cross_entropy(logits, Y_test)
print(test_loss.item(), "\n")

for i in range(20):

    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + B1)
        logits = h @ W2 + B2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(itos[letter] for letter in out))
end_time = time.time()
t = end_time-start_time
print("Time Taken:", t/60)
