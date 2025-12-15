import torch
import torch.nn as nn

class MultiTimeframeFusion(nn.Module):
    """多時間框架融合模型 - 1h 和 15m 互相調教"""
    
    def __init__(self, input_size=44, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        
        # 1h 時間框架編碼器
        self.encoder_1h = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # 15m 時間框架編碼器
        self.encoder_15m = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # 交叉注意力層 - 讓 1h 關注 15m 的細節
        self.cross_attention_1h_to_15m = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=dropout
        )
        
        # 交叉注意力層 - 讓 15m 關注 1h 的趨勢
        self.cross_attention_15m_to_1h = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=dropout
        )
        
        # 融合層
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 輸出層
        self.fc_output = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x_1h, x_15m):
        """
        x_1h: (batch, seq_len, 44) - 1h 數據
        x_15m: (batch, seq_len * 4, 44) - 15m 數據（4 倍數量）
        """
        # 編碼各自時間框架
        gru_1h, _ = self.encoder_1h(x_1h)      # (batch, seq_len, hidden)
        gru_15m, _ = self.encoder_15m(x_15m)   # (batch, seq_len*4, hidden)
        
        # 1h 學習 15m 的細節（短期波動）
        enhanced_1h, _ = self.cross_attention_1h_to_15m(
            gru_1h,           # query: 1h 想學什麼
            gru_15m,          # key, value: 15m 提供細節
            gru_15m
        )
        
        # 15m 學習 1h 的趨勢（長期方向）
        enhanced_15m, _ = self.cross_attention_15m_to_1h(
            gru_15m,         # query: 15m 想學什麼
            gru_1h,          # key, value: 1h 提供趨勢
            gru_1h
        )
        
        # 取最後一個時間步
        last_1h = enhanced_1h[:, -1, :]      # (batch, hidden)
        last_15m = enhanced_15m[:, -1, :]    # (batch, hidden)
        
        # 融合兩個時間框架
        fused = torch.cat([last_1h, last_15m], dim=1)  # (batch, hidden*2)
        fused_out = self.fusion(fused)                  # (batch, hidden//2)
        
        # 預測
        output = self.fc_output(fused_out)              # (batch, 1)
        return output


if __name__ == '__main__':
    # 測試模型
    model = MultiTimeframeFusion()
    x_1h = torch.randn(32, 60, 44)      # 32 batch, 60 time steps, 44 features
    x_15m = torch.randn(32, 240, 44)    # 32 batch, 240 time steps, 44 features
    output = model(x_1h, x_15m)
    print(f'Output shape: {output.shape}')  # Should be (32, 1)
