
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool


class SequenceCNN(nn.Module):

    def __init__(self, vocab_size=21, embed_dim=128, num_filters=128, filter_sizes=[3, 5, 7], output_dim=128):
        super(SequenceCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes
        ])

        self.projection = nn.Linear(embed_dim, len(filter_sizes) * num_filters) if embed_dim != len(
            filter_sizes) * num_filters else None

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):

        embedded = self.embedding(x)
        original_embedded = embedded.mean(dim=1)

        embedded = embedded.permute(0, 2, 1)

        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)


        if self.projection is not None:
            residual = self.projection(original_embedded)
        else:
            residual = original_embedded

        cat = cat + residual
        dropout = self.dropout(cat)
        output = self.fc(dropout)
        output = self.layer_norm(output)

        return output


class AttentionFusion(nn.Module):

    def __init__(self, graph_feature_dim=128, seq_feature_dim=128):
        super(AttentionFusion, self).__init__()
        self.graph_feature_dim = graph_feature_dim
        self.seq_feature_dim = seq_feature_dim
        # 注意力权重计算
        self.attention_weight = nn.Linear(graph_feature_dim + seq_feature_dim, 2)

    def forward(self, graph_feature, seq_feature):

        concat_features = torch.cat([graph_feature, seq_feature], dim=1)
        attention_scores = F.softmax(self.attention_weight(concat_features), dim=1)
        weighted_graph = graph_feature * attention_scores[:, 0:1]
        weighted_seq = seq_feature * attention_scores[:, 1:2]
        fused_features = torch.cat([weighted_graph, weighted_seq], dim=1)
        return fused_features


class EquivariantGNNLayer(torch.nn.Module):


    def __init__(self, in_size, out_size):
        super(EquivariantGNNLayer, self).__init__()
        from torch_geometric.nn import GCNConv
        self.conv = GCNConv(in_size, out_size)
        self.layer_norm = nn.LayerNorm(out_size)

        if in_size != out_size:
            self.residual_projection = nn.Linear(in_size, out_size)
        else:
            self.residual_projection = None

        self.edge_nn = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, graph, node_feat, coord_feat):
        original_node_feat = node_feat
        edge_attr = self.compute_edge_attributes(graph.edge_index, coord_feat)
        edge_weights = self.edge_nn(edge_attr).squeeze(-1) if edge_attr.size(0) > 0 else None
        new_node_feat = self.conv(node_feat, graph.edge_index, edge_weights)
        if self.residual_projection is not None:
            residual = self.residual_projection(original_node_feat)
        else:
            residual = original_node_feat
        new_node_feat = new_node_feat + residual
        new_node_feat = self.layer_norm(new_node_feat)
        new_coord_feat = coord_feat
        return new_node_feat, new_coord_feat

    def compute_edge_attributes(self, edge_index, coord_feat):
        if edge_index.size(1) == 0:
            return torch.empty(0, 4, device=coord_feat.device)
        row, col = edge_index
        diff = coord_feat[row] - coord_feat[col]
        dist = torch.norm(diff, dim=1, keepdim=True)
        edge_attr = torch.cat([diff, dist], dim=1)  # [dx, dy, dz, distance]
        return edge_attr

class EquivariantPPI(torch.nn.Module):

    def __init__(self, args):
        super(EquivariantPPI, self).__init__()
        torch.backends.cudnn.enabled = False
        self.type = args['task_type']
        self.embedding_size = args['emb_dim']
        self.hidden_size = args['hidden_dim'] if 'hidden_dim' in args else 256
        self.output_dim = args['output_dim']
        self.num_layers = args.get('num_layers', 3)
        self.egnn_layers = nn.ModuleList()
        self.egnn_layers.append(EquivariantGNNLayer(self.embedding_size, self.hidden_size))
        for _ in range(self.num_layers - 1):
            self.egnn_layers.append(EquivariantGNNLayer(self.hidden_size, self.hidden_size))
        self.node_to_graph = nn.Linear(self.hidden_size, self.output_dim)
        self.peptide_cnn = SequenceCNN()
        self.protein_cnn = SequenceCNN()
        self.fusion_layer = AttentionFusion(graph_feature_dim=self.output_dim, seq_feature_dim=128)
        fusion_output_dim = 512
        self.fc1 = nn.Linear(fusion_output_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.1)
        self.skip_fc = nn.Linear(1024, 256)

        self.out = nn.Linear(128, 1)

        self.relu = nn.ReLU()

        self._initialize_weights()

    def forward_graph(self, graph, initial_features):
        num_nodes = graph.num_nodes
        if num_nodes == 0:
            batch_size = graph.num_graphs
            return torch.zeros(batch_size, self.output_dim).to(initial_features.device)
        coord_features = torch.randn(num_nodes, 3).to(initial_features.device)
        node_features = initial_features
        node_features = nn.LayerNorm(node_features.size(-1)).to(node_features.device)(node_features)
        for egnn_layer in self.egnn_layers:
            node_features, coord_features = egnn_layer(graph, node_features, coord_features)
        graph_features = global_max_pool(node_features, graph.batch)
        graph_features = self.relu(self.node_to_graph(graph_features))

        return graph_features

    def forward(self, receptor_graph, peptide_graph, receptor_seq, peptide_seq):

        device = next(self.parameters()).device
        receptor_seq = receptor_seq.to(device)
        peptide_seq = peptide_seq.to(device)

        receptor_graph_feat = self.forward_graph(receptor_graph, receptor_graph.x)

        peptide_graph_feat = self.forward_graph(peptide_graph, peptide_graph.x)

        receptor_seq_feat = self.protein_cnn(receptor_seq)

        peptide_seq_feat = self.peptide_cnn(peptide_seq)

        fused_receptor = self.fusion_layer(receptor_graph_feat, receptor_seq_feat)

        fused_peptide = self.fusion_layer(peptide_graph_feat, peptide_seq_feat)

        combined_features = torch.cat([fused_receptor, fused_peptide], dim=1)

        x = self.fc1(combined_features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        skip_connection = x

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        # 添加跳跃连接
        skip_proj = self.skip_fc(skip_connection)
        x = x + skip_proj
        x = self.relu(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)

        # 输出层
        out = self.out(x)
        output = torch.sigmoid(out)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)