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

        # Calculate the div_term for the positional encoding function (2i) and (2i+1) indices separately
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

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """
        Initialize the LayerNormalization module.

        Args:
            d_model (int): The dimensionality of the model.
            eps (float): A small value to prevent division by zero.

        Returns:
            None
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1)) # Learnable scale - Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Learnable bias - Added

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LayerNormalization module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized input tensor.
        """
        # Calculate the mean of the input tensor over the last dimension (feature dimension) and keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        
        # Calculate the standard deviation of the input tensor over the last dimension (feature dimension) and keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)

        # Normalize the input tensor and return it with the learnable scale and bias applied
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    FeedForwardBlock is a module that applies a feed-forward neural network to the input tensor.

    Args:
        d_model (int): The number of expected features in the input tensor.
        d_ff (int): The number of output features in the feed-forward neural network.
        dropout (float): The probability of an element to be zeroed.

    Attributes:
        linear_1 (nn.Linear): The first linear layer of the feed-forward neural network.
        dropout (nn.Dropout): The dropout layer.
        linear_2 (nn.Linear): The second linear layer of the feed-forward neural network.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeedForwardBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the feed-forward neural network.
        """
        # (Batch, Seq, d_model) -> (Batch, Seq, d_ff) -> (Batch, Seq, d_model)
        
        # Apply ReLU to the output of the first linear layer and then apply dropout to the output of the ReLU and then apply the second linear layer
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention is a module that applies multi-head attention to the input tensor.
    
    Args:
        d_model (int): The number of expected features in the input tensor.
        num_heads (int): The number of heads in the multi-head attention.
        dropout (float): The probability of an element to be zeroed.
        
        Attributes:
        d_model (int): The number of expected features in the input tensor.
        
        linear_q (nn.Linear): The linear layer for the query matrix.
        linear_k (nn.Linear): The linear layer for the key matrix.
        linear_v (nn.Linear): The linear layer for the value matrix.
        linear_out (nn.Linear): The linear layer to project the concatenated output of all heads.
        dropout (nn.Dropout): The dropout layer.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Check if the d_model is divisible by the number of heads
        assert d_model % num_heads == 0, "The number of heads must be divisible by the d_model"

        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

        # Calculate the attention scores
        @staticmethod
        def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, dropout: nn.Dropout = None) -> torch.Tensor:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            scores = F.softmax(scores, dim=-1)

            if dropout is not None:
                scores = dropout(scores)

            return torch.matmul(scores, value)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        query = self.w_q(query) # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
        key = self.w_k(key) # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
        value = self.w_v(value) # (Batch, Seq, d_model) -> (Batch, Seq, d_model)

        # Split the query, key, and value into multiple heads
        # (Batch, Seq, d_model) -> (Batch, Seq, num_heads, d_k) -> (Batch, num_heads, Seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # Apply the attention mechanism to the query, key, and value
        x = self.attention(query, key, value, mask, self.dropout)

        # Concatenate the output of all heads
        # (Batch, num_heads, Seq, d_k) -> (Batch, Seq, num_heads, d_k) -> (Batch, Seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k) # d_model = num_heads * d_k

        # Apply the output linear layer
        return self.w_o(x)

