"""
LSTM Move Classifier Model
===========================
Three-head shared-backbone LSTM for fighting game move classification.

Architecture:
  Input (53 features) → LayerNorm → LSTM(2 layers, 64 hidden) → Dropout
    → Upper body head (4 classes: idle, left_punch, right_punch, block)
    → Lower body head (5 classes: idle, left_kick, right_kick, crouch, jump)
    → Movement head   (3 classes: idle, move_left, move_right)

The three heads allow independent classification — a punch CAN co-occur
with move_right, but a punch and block cannot.
"""

import torch
import torch.nn as nn


# Class name mappings for each head
UPPER_CLASSES = ["idle_upper", "left_punch", "right_punch", "block"]
LOWER_CLASSES = ["idle_lower", "left_kick", "right_kick", "crouch", "jump"]
MOVEMENT_CLASSES = ["idle_movement", "move_left", "move_right"]

# Mapping from recording label (integer 0-9) to per-head targets
# Recording labels: 0=idle,1=l_punch,2=r_punch,3=block,4=l_kick,5=r_kick,6=crouch,7=jump,8=move_left,9=move_right
LABEL_TO_HEADS = {
    0: (0, 0, 0),  # idle → all heads idle
    1: (1, 0, 0),  # left_punch → upper=1
    2: (2, 0, 0),  # right_punch → upper=2
    3: (3, 0, 0),  # block → upper=3
    4: (0, 1, 0),  # left_kick → lower=1
    5: (0, 2, 0),  # right_kick → lower=2
    6: (0, 3, 0),  # crouch → lower=3
    7: (0, 4, 0),  # jump → lower=4
    8: (0, 0, 1),  # move_left → movement=1
    9: (0, 0, 2),  # move_right → movement=2
}

# Reverse mapping: per-head class index → move name string (for inference output)
UPPER_TO_MOVE = {1: "left_punch", 2: "right_punch", 3: "block"}
LOWER_TO_MOVE = {1: "left_kick", 2: "right_kick", 3: "crouch", 4: "jump"}
MOVEMENT_TO_MOVE = {1: "move_left", 2: "move_right"}


class ThreeHeadLSTM(nn.Module):
    """
    Three-head LSTM for fighting move classification.

    Args:
        input_size:  Number of features per frame (default 53).
        hidden_size: LSTM hidden state size (default 64).
        num_layers:  Stacked LSTM layers (default 2).
        dropout:     Dropout rate between LSTM layers and before heads (default 0.3).
    """

    def __init__(self, input_size=53, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layer_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Classification heads
        self.upper_head = nn.Linear(hidden_size, len(UPPER_CLASSES))    # 4
        self.lower_head = nn.Linear(hidden_size, len(LOWER_CLASSES))    # 5
        self.movement_head = nn.Linear(hidden_size, len(MOVEMENT_CLASSES))  # 3

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
            hidden: Optional (h0, c0) tuple. If None, zeros are used.

        Returns:
            upper_logits: (batch, 4)
            lower_logits: (batch, 5)
            movement_logits: (batch, 3)
            hidden: Updated (h, c) tuple for stateful inference.
        """
        batch_size = x.size(0)

        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # Normalize input features
        x = self.layer_norm(x)

        # LSTM: process full sequence
        lstm_out, hidden = self.lstm(x, hidden)

        # Take the output of the last timestep
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        last_output = self.dropout(last_output)

        # Three independent classification heads
        upper_logits = self.upper_head(last_output)
        lower_logits = self.lower_head(last_output)
        movement_logits = self.movement_head(last_output)

        return upper_logits, lower_logits, movement_logits, hidden

    def init_hidden(self, batch_size, device):
        """Create zeroed initial hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)
