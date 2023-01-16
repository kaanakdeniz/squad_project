import torch

def calc_multiplicative_attention(weight, x):
    batch_size, seq_len, input_size = x.size()
    Wx = weight(x)  # (batch_size, seq_len, input_size)
    x_copy = torch.transpose(x, 1, 2)  # (batch_size, input_size, seq_len)

    s = torch.bmm(Wx, x_copy)  # (batch_size, seq_len, seq_len)
    mask = (torch.eye(seq_len) == 1).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    s.masked_fill_(mask, float('-inf'))  # mask out similarity between the same tokens
    a = torch.softmax(s, dim=2)  # (batch_size, seq_len, seq_len)
    return torch.bmm(a, x)


def calc_gated_attention(x, c, gate, rnn, proj=None):
    rnn_in = torch.cat([x, c], dim=2)       # (batch_size, seq_len, 2 * input_size)
    rnn_in = rnn_in * gate(rnn_in)      # (batch_size, seq_len, 2 * input_size)
    h, _ = rnn(rnn_in)  # (batch_size, seq_len, 2 * hidden_size)
    return h if proj is None else proj(h)   # (batch_size, seq_len, input_size)


def calc_additive_attention(W1, W2, v, x):
    batch_size, seq_len, input_size = x.size()
    W1x = W1(x)  # (batch_size, seq_len, att_dim)
    W2x = W2(x)  # (batch_size, seq_len, att_dim)

    s = torch.zeros((batch_size, seq_len, seq_len)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # process one sequence at a time to save memory
    for idx in range(seq_len):
        s[:, idx, :] = v(torch.tanh(W1x + W2x[:, idx, :].unsqueeze(1))).transpose(1, 2).squeeze()  # (batch_size, 1, seq_len)
    mask = (torch.eye(seq_len) == 1).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    s.masked_fill_(mask, float('-inf'))  # mask out similarity between the same tokens

    a = torch.softmax(s, dim=2)  # (batch_size, seq_len, seq_len)
    return torch.bmm(a, x)
