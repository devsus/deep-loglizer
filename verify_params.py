import torch.nn as nn

lstm = nn.LSTM(
            input_size=32,  # embedding dim
            hidden_size=128,
            batch_first=True,
            num_layers=2,
            bidirectional=True,     # if dir=2
        )

total = sum(p.numel() for p in lstm.parameters())
print(total)

encoder_layer = nn.TransformerEncoderLayer(
    d_model=32,             # embedding_dim
    nhead=2,                # heads scaled for hidden size
    dim_feedforward=128,    # hidden size
    batch_first=True
)
transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
total = sum(p.numel() for p in transformer.parameters())
print(total)