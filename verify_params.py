import torch.nn as nn

lstm = nn.LSTM(
            input_size=512,  # embedding dim
            hidden_size=2048,
            batch_first=True,
            num_layers=32,
            bidirectional=True,     # if dir=2
        )

total = sum(p.numel() for p in lstm.parameters())
print(total)

encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,             # embedding_dim
    nhead=32,                # heads scaled for hidden size
    dim_feedforward=2048,    # hidden size
    batch_first=True
)
transformer = nn.TransformerEncoder(encoder_layer, num_layers=32)
total = sum(p.numel() for p in transformer.parameters())
print(total)