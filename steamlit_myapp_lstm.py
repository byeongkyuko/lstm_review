import streamlit as st
import torch
import spacy
from torchtext.legacy.data import Field
import torch
import torch.nn as nn
import torch.optim as optim

# 모델 정의 (훈련과 동일하게)
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)

# 모델 초기화 및 가중치 로드
model = LSTMModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
model.load_state_dict(torch.load('lstm_model.pth', map_location=torch.device('cpu')))
model.eval()

# Streamlit 웹 애플리케이션
st.title('LSTM Sentiment Analysis')

# 사용자 입력 받기
user_input = st.text_area('Enter your text here:')

if st.button('Predict'):
    if user_input:
        # 입력 텍스트 전처리
        nlp = spacy.load('en_core_web_sm')
        tokenized = [tok.text for tok in nlp.tokenizer(user_input)]
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1).T
        length_tensor = torch.LongTensor(length)
        
        # 예측 수행
        prediction = torch.sigmoid(model(tensor, length_tensor))
        result = 'Positive' if prediction.item() > 0.5 else 'Negative'
        
        # 예측 결과 출력
        st.write('Prediction:', result)
    else:
        st.write('Please enter some text.')
