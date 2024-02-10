import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the InputEmbedding module.

        Args:
            d_model (int): The dimensionality of the embedding.
            vocab_size (int): The size of the vocabulary.

        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Perform the forward pass of the InputEmbedding module.

        Args:
            x: The input tensor.

        Returns:
            torch.Tensor: The embedded input tensor.

        """
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, seq_len: int):
        """
        Initialize the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the model.
            dropout (float): The dropout rate.
            seq_len (int): The length of the sequence.

        Returns:
            None
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.seq_len = seq_len

        # Create a matrix shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len - 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # Add a batch dimension

        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Forward pass of the PositionalEncoding module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor with positional encoding added.
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Add positional encoding to the input
        return self.dropout(x)