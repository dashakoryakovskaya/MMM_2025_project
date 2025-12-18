# Импорт необходимых библиотек
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, RGCNConv, TransformerConv


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.gelu(x)  # Более плавная активация
        x = self.dropout(x)
        return self.layer_2(x)


class AddAndNorm(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)].detach()  # Отключаем градиенты
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1, positional_encoding=False):
        super().__init__()
        self.input_dim = input_dim
        self.self_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionWiseFeedForward(input_dim, input_dim, dropout=dropout)
        self.add_norm_after_attention = AddAndNorm(input_dim, dropout=dropout)
        self.add_norm_after_ff = AddAndNorm(input_dim, dropout=dropout)
        self.positional_encoding = PositionalEncoding(input_dim) if positional_encoding else None

    def forward(self, key, value, query):
        if self.positional_encoding:
            key = self.positional_encoding(key)
            value = self.positional_encoding(value)
            query = self.positional_encoding(query)

        attn_output, _ = self.self_attention(query, key, value, need_weights=False)

        x = self.add_norm_after_attention(attn_output, query)

        ff_output = self.feed_forward(x)
        x = self.add_norm_after_ff(ff_output, x)

        return x


class GraphFusionLayerAtt(nn.Module):
    def __init__(self, hidden_dim, heads=2):
        super().__init__()
        # Проекционные слои для признаков
        self.proj_audio = nn.Linear(hidden_dim, hidden_dim)
        self.proj_text = nn.Linear(hidden_dim, hidden_dim)

        # Графовые слои
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim*heads, hidden_dim)

        self.attention_fusion = nn.Linear(hidden_dim, 1)

        # Финальная проекция
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def build_complete_graph(self, num_nodes):
        # Создаем полный граф (каждый узел соединен со всеми)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index).t().contiguous()

    def forward(self, first_stats, second_stats):
        """
        first_stats: [batch_size, hidden_dim]
        second_stats: [batch_size, hidden_dim]
        """
        batch_size = first_stats.size(0)

        # Проекция признаков
        x_first = F.relu(self.proj_audio(first_stats))  # [batch_size, hidden_dim]
        x_second = F.relu(self.proj_text(second_stats))    # [batch_size, hidden_dim]

        # Объединение узлов (1 и 2 попеременно)
        nodes = torch.stack([x_first, x_second], dim=1)  # [batch_size, 2, hidden_dim]
        nodes = nodes.view(-1, nodes.size(-1))        # [batch_size*2, hidden_dim]

        # Построение графа (полный граф для каждого элемента батча)
        edge_index = self.build_complete_graph(2)  # Граф для одной пары 1-2
        edge_index = edge_index.to(first_stats.device)

        # Применение GAT
        x = F.relu(self.gat1(nodes, edge_index))
        x = self.gat2(x, edge_index)

        # Разделяем обратно аудио и текст
        x = x.view(batch_size, 2, -1)  # [batch_size, 2, hidden_dim]

        # Усреднение по модальностям
        # fused = torch.mean(x, dim=1)   # [batch_size, hidden_dim]

        weights = F.softmax(self.attention_fusion(x), dim=1)
        fused = torch.sum(weights * x, dim=1)  # [batch_size, hidden_dim]

        return self.fc(fused)


class MultiModalTransformer(nn.Module):
    def __init__(self, first_dim=768, second_dim=1024, hidden_dim=512, num_transformer_heads=2, positional_encoding=True, dropout=0, mode='mean', device="cuda",  tr_layer_number=1, out_features=128, num_heads=2):
        super(MultiModalTransformer, self).__init__()

        self.mode = mode

        self.hidden_dim = hidden_dim

        # Проекционные слои

        self.first_proj = nn.Sequential(
            nn.Conv1d(first_dim, hidden_dim, 1),
            nn.GELU(),
        )

        self.second_proj = nn.Sequential(
            nn.Conv1d(second_dim, hidden_dim, 1),
            nn.GELU(),
        )

        # Механизмы внимания
        self.first_to_second_attn = nn.ModuleList([TransformerEncoderLayer(input_dim=hidden_dim, num_heads=num_transformer_heads, positional_encoding=positional_encoding, dropout=dropout) for i in range(tr_layer_number)
                ])
        self.second_to_first_attn = nn.ModuleList([TransformerEncoderLayer(input_dim=hidden_dim, num_heads=num_transformer_heads, positional_encoding=positional_encoding, dropout=dropout) for i in range(tr_layer_number)
                ])
        
        # Графовое слияние
        if self.mode == 'mean':
            self.graph_fusion = GraphFusionLayerAtt(hidden_dim, heads=num_heads)
        else:
            self.graph_fusion = GraphFusionLayerAtt(hidden_dim*2, heads=num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, out_features) if self.mode == 'mean' else nn.Linear(hidden_dim*2, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, 1)
        )

    def forward(self, first_features, second_features):
        # Преобразование размерностей
        first_features = first_features.float()
        second_features = second_features.float()

        first_features = self.first_proj(first_features.permute(0,2,1)).permute(0,2,1)
        second_features = self.second_proj(second_features.permute(0,2,1)).permute(0,2,1)

        # Адаптивная пуллинг до минимальной длины
        min_seq_len = min(first_features.size(1), second_features.size(1))
        first_features = F.adaptive_avg_pool1d(first_features.permute(0,2,1), min_seq_len).permute(0,2,1)
        second_features = F.adaptive_avg_pool1d(second_features.permute(0,2,1), min_seq_len).permute(0,2,1)

        # Трансформерные блоки
        for i in range(len(self.first_to_second_attn)):
            attn_first = self.first_to_second_attn[i](second_features, first_features, first_features)
            attn_second = self.second_to_first_attn[i](first_features, second_features, second_features)
            first_features += attn_first
            second_features += attn_second

        # Статистики
        std_first, mean_first = torch.std_mean(attn_first, dim=1)
        std_second, mean_second = torch.std_mean(attn_second, dim=1)

        # Графовое слияние статистик
        if self.mode == 'mean':
            h_ta = self.graph_fusion(mean_first, mean_second)
        else:
            std_first = torch.nan_to_num(std_first, nan=0.0)
            std_second = torch.nan_to_num(std_second, nan=0.0)
            h_ta = self.graph_fusion(torch.cat([mean_first, std_first], dim=1), torch.cat([mean_second, std_second], dim=1))

        # Классификация
        return self.classifier(h_ta)
