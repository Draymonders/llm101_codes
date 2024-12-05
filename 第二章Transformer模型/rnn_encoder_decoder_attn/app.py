import streamlit as st
import torch
from seq2seq_translation_tutorial import (
    EncoderRNN,
    AttnDecoderRNN,
    Vocab,
    evaluate,
    device,
)

# 页面配置
st.set_page_config(page_title="中英翻译", layout="wide")

@st.cache_resource
def load_model():
    # 加载词汇表
    input_vocab = Vocab.load("input.vocab")
    output_vocab = Vocab.load("output.vocab")
    
    # 初始化模型
    hidden_size = 128
    encoder = EncoderRNN(input_vocab.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_vocab.n_words).to(device)
    
    # 加载模型权重
    if torch.cuda.is_available():
        checkpoint = torch.load("checkpoint.pt")
    else:
        checkpoint = torch.load("checkpoint.pt", map_location=torch.device("cpu"))
        
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, input_vocab, output_vocab

def translate(text, encoder, decoder, input_vocab, output_vocab):
    output_words, attentions = evaluate(
        encoder, decoder, text, input_vocab, output_vocab
    )
    return " ".join(output_words)

def main():
    st.title("基于RNN Encoder-Decoder with Attention的中英翻译系统")
    
    # 加载模型
    try:
        encoder, decoder, input_vocab, output_vocab = load_model()
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return
    
    # 创建输入框
    input_text = st.text_area("请输入中文:", height=100)
    
    if st.button("翻译"):
        if input_text:
            try:
                translation = translate(input_text, encoder, decoder, input_vocab, output_vocab)
                st.success("翻译结果:")
                st.write(translation.replace("<EOS>", ""))
            except Exception as e:
                st.error(f"翻译过程出错: {str(e)}")
        else:
            st.warning("请输入要翻译的文本")

if __name__ == "__main__":
    main() 