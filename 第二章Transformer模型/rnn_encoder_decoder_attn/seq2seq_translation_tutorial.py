import time
import math
import argparse
from io import open
import random
import string

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import joblib
import pkuseg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0  # 输入给decoder的第一个token，表示句子开始
EOS_token = 1  # 句子结束符
UNK_token = 2  # 未登录词
MAX_LENGTH = 10


class Vocab:
    def __init__(self):
        self.word2index = {}  # {word: word_indx}
        self.word2count = {}  # {word: word_freq}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS, EOS and UNK

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def save(self, path):
        joblib.dump(
            {
                "word2index": self.word2index,
                "word2count": self.word2count,
                "index2word": self.index2word,
                "n_words": self.n_words,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        vocab = cls()
        checkpoint = joblib.load(path)
        vocab.word2index = checkpoint["word2index"]
        vocab.word2count = checkpoint["word2count"]
        vocab.index2word = checkpoint["index2word"]
        vocab.n_words = checkpoint["n_words"]
        return vocab

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def remove_punctuation_regex(text):
    """
    去除所有中英文标点符号
    """
    # 定义中文标点符号
    chinese_punc = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛"
        "„‟…‧﹏"
    )

    # 先去除英文标点
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 再去除中文标点
    text = text.translate(str.maketrans("", "", chinese_punc))

    return text


def readLangs(data_path):
    print("Reading lines...")

    # 用pkuseg进行中文分词
    seg = pkuseg.pkuseg()

    # Read the file and split into lines
    with open(data_path, encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    pairs = []
    for line in lines:
        en, zh, _ = line.split("\t")
        en = remove_punctuation_regex(en.lower().strip())
        zh = remove_punctuation_regex(zh.strip())
        zh = " ".join(seg.cut(zh))
        # print(en, zh)  # youll never be alone 你 永远 不 会 一个 人 的
        # print(type(zh), type(en))  # <class 'str'> <class 'str'>

        if (
            len(zh.split(" ")) < MAX_LENGTH
            and len(en.split(" ")) < MAX_LENGTH
            and len(zh) > 0
            and len(en) > 0
        ):
            pairs.append([zh, en])
    input_vocab = Vocab()
    output_vocab = Vocab()
    return input_vocab, output_vocab, pairs


def prepareData(data_path):
    input_vocab, output_vocab, pairs = readLangs(data_path)
    print(f"Read {len(pairs)} sentence pairs")
    print("Counting words...")
    for pair in pairs:  # 构建词表
        input_vocab.addSentence(pair[0])
        output_vocab.addSentence(pair[1])
    print(f"Counted words: {input_vocab.n_words}, {output_vocab.n_words}")
    return input_vocab, output_vocab, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)  # batch sequence, [Batch_size, Sequence_length, hidden_size]
        # [sequence_length, batch_size, hidden_size]
        # [1,2,3,0,0]
        # [1,1,1,1,0]
        # [1,1,1,1,1]
        # t=1, t=2, t=3 , t=4, t=5        

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, input_lengths):
        embedded = self.dropout(self.embedding(input))
        
        # [1,1,1,0,0]
        # [1,1,1,1,0]
        # [1,1,1,1,1]

        # pack变长序列
        packed_input = pack_padded_sequence(
            embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, hidden = self.gru(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys, mask=None):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        # print(f"scores shape: {scores.shape}")  # torch.Size([B, L, 1])
        scores = scores.squeeze(2).unsqueeze(1)  # [B, 1, L]

        if mask is not None:
            scores.data.masked_fill_(mask == 0, -float("inf"))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)  # [B, 1, L] @ [B, L, 128] -> [B, 1, 128]
        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, mask=None):
        # print(f"encoder_outputs shape: {encoder_outputs.shape}")  # torch.Size([B, L, D(128)])
        # print(f"encoder_hidden shape: {encoder_hidden.shape}")  # torch.Size([1, B, 128])
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(  # [B, 1]
            batch_size, 1, dtype=torch.long, device=device
        ).fill_(SOS_token)
        # 初始化decoder的hidden状态使用encoder最后一个时刻的hidden状态
        decoder_hidden = encoder_hidden  # [1, B, D(128)]
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):  # 遍历decoder的每一个时刻t
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, mask
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs, mask=None):
        embedded = self.dropout(self.embedding(input))
        # print(f"hidden shape: {hidden.shape}")  # torch.Size([1, B, 128])
        query = hidden.permute(1, 0, 2)  # [B, 1, 128]
        context, attn_weights = self.attention(query, encoder_outputs, mask)
        # print(f"context shape: {context.shape}")  # torch.Size([B, 1, 128])
        input_gru = torch.cat((embedded, context), dim=2)  # [B, 1, 256]

        # 保留GRU结构
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        # print(f"output shape: {output.shape}")  # torch.Size([B, 1, V])
        # print(f"hidden shape: {hidden.shape}")  # torch.Size([1, B, 128])

        return output, hidden, attn_weights


def indexesFromSentence(vocab, sentence):
    return [
        vocab.word2index[word] if word in vocab.word2index else vocab.word2index["UNK"]
        for word in sentence.split(" ")
    ]


def tensorFromSentence(vocab, sentence):
    seg = pkuseg.pkuseg()
    zh = remove_punctuation_regex(sentence.strip())
    zh = " ".join(seg.cut(zh))
    indexes = indexesFromSentence(vocab, zh)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def get_dataloader(data_path, batch_size):
    input_vocab, output_vocab, pairs = prepareData(data_path)  # i like eating 我 喜欢 吃

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)  # padding
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)  # padding

    input_lengths = np.zeros(n, dtype=np.int32)  # 记录源语言序列长度
    target_lengths = np.zeros(n, dtype=np.int32)  # 记录目标语言序列长度

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_vocab, inp)
        tgt_ids = indexesFromSentence(output_vocab, tgt)  # 56 6 23
        inp_ids.append(EOS_token)  # 添加句子结束符
        tgt_ids.append(EOS_token)  # 添加句子结束符  # 56 6 23 1
        input_lengths[idx] = len(inp_ids)  # 记录源语言序列实际长度
        target_lengths[idx] = len(tgt_ids)  # 记录目标语言序列实际长度
        input_ids[idx, : len(inp_ids)] = inp_ids
        target_ids[idx, : len(tgt_ids)] = tgt_ids
    print(input_ids[:2])
    print(target_ids[:2])

    # 创建包含长度信息的数据集
    train_data = TensorDataset(
        torch.LongTensor(input_ids).to(device),
        torch.LongTensor(target_ids).to(device),
        torch.LongTensor(input_lengths).to(device),
        torch.LongTensor(target_lengths).to(device),
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )
    return input_vocab, output_vocab, train_dataloader


def train_epoch(
    dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion
):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor, input_lengths, target_lengths = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # 创建 attention mask
        # mask shape: [batch_size, 1, max_length]
        mask = torch.arange(MAX_LENGTH).expand(input_tensor.size(0), MAX_LENGTH).to(
            device
        ) < input_lengths.unsqueeze(1)
        mask = mask.unsqueeze(1)  # 为attention的broadcast添加维度
        # [1 1 1 0 0 ]
        # [1 1 1 1 0] 

        encoder_outputs, encoder_hidden = encoder(
            input_tensor, input_lengths
        )  # encoding src word sequence
        
        # print(encoder_outputs.shape, encoder_hidden.shape)  # torch.Size([B, L, 128]) torch.Size([1, B, 128])
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        # 计算损失时只考虑有效位置
        loss = 0
        for b in range(target_tensor.size(0)):
            loss += criterion(
                decoder_outputs[b, : target_lengths[b]],
                target_tensor[b, : target_lengths[b]],
            )
        loss = loss / target_tensor.size(0)  # batch mean
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def evaluate(encoder, decoder, sentence, input_vocab, output_vocab):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_vocab, sentence)
        # print(f"input_tensor shape: {input_tensor.shape}")  # torch.Size([1, L])
        input_lengths = torch.LongTensor([input_tensor.size(1)]).to(device)

        encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden
        )

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_vocab.index2word[idx.item()])
    return decoded_words, decoder_attn


def train(
    train_dataloader,
    encoder,
    decoder,
    n_epochs,
    learning_rate,
    print_every=100,
):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    best_loss = float("inf")
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(
            train_dataloader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        print_loss_total += loss

        # Save best model
        if loss < best_loss:
            best_loss = loss
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                },
                "checkpoint.pt",
            )

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                f"{timeSince(start, epoch / n_epochs)} ({epoch} {epoch / n_epochs * 100}%) {print_loss_avg:.4f}"
            )


def load_checkpoint(encoder, decoder, checkpoint_path="checkpoint.pt"):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])


def main():
    parser = argparse.ArgumentParser(description="Seq2Seq Translation Model")
    
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--n_epochs", type=int, default=20, help="Number of epoch for training"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Equals to word embedding size and hidden size of GRU",
    )
    parser.add_argument(
        "--data_path", type=str, default="cmn.txt", help="Path to data file"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoint.pt",
        help="Path to saved model checkpoint",
    )
    parser.add_argument("--evaluate", action="store_true", help="是否测试训练好的模型")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    set_seed(args.seed)

    if not args.evaluate:
        input_vocab, output_vocab, pairs = prepareData(args.data_path)
        # 保存源语言和目标语言的词汇表
        input_vocab.save("input.vocab")
        output_vocab.save("output.vocab")
        print(random.choice(pairs))

        input_vocab, output_vocab, train_dataloader = get_dataloader(
            args.data_path, args.batch_size
        )
        # class RNNEncoderDecodr(nn.Module)
        # current batch, input data --> computation graph --> output data
        # input data --> EncoderRNN --> AttnDecoderRNN --> output data
        # end2end model
        encoder = EncoderRNN(input_vocab.n_words, args.hidden_size, args.dropout).to(
            device
        )
        decoder = AttnDecoderRNN(
            args.hidden_size, output_vocab.n_words, args.dropout
        ).to(device)

        train(
            train_dataloader,
            encoder,
            decoder,
            args.n_epochs,
            args.learning_rate,
            print_every=5,
        )
    else:
        # 加载源语言和目标语言的词汇表
        input_vocab = Vocab.load("input.vocab")
        output_vocab = Vocab.load("output.vocab")
        encoder = EncoderRNN(input_vocab.n_words, args.hidden_size).to(device)
        decoder = AttnDecoderRNN(args.hidden_size, output_vocab.n_words).to(device)
        load_checkpoint(encoder, decoder)

        encoder.eval()
        decoder.eval()
        input_sentence = "你好明天"
        output_words, attentions = evaluate(
            encoder, decoder, input_sentence, input_vocab, output_vocab
        )
        print(f"input = {input_sentence}")
        print(f"output = {' '.join(output_words)}")


if __name__ == "__main__":
    main()
