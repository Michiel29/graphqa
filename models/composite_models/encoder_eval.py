from models.composite_models.encoder_gnn import EncoderGNNModel

class EncoderGNNEval(EncoderGNNModel):

    def forward(self, batch):
        text_enc = self.encode_text(batch['text'])
        device = text_enc.device

        target_text_idx = batch['target_text_idx']
        n_targets = len(target_text_idx)
        graph_idx = batch['graph']
        graph_sizes = batch['graph_sizes']

        target_text_idx = target_text_idx.reshape(-1) # (n_targets)

        target_idx_range = torch.arange(n_targets, device=device) # (n_targets)
        put_indices = tuple(torch.repeat_interleave(target_idx_range, graph_sizes, dim=0).unsqueeze(0)) # (sum(m_i))

        graph_rep = text_enc[graph_idx] # (sum(m_i), 2, d)
        target_rep = text_enc[target_text_idx] # (n_targets, d)

        if self.args.placeholder_input:
            layer_target_rep = self.placeholder_input.unsqueeze(0).expand(n_targets, -1)
        else:
            layer_target_rep = target_rep

        for layer in self.gnn_layers:
            layer_target_rep, graph_rep = layer(layer_target_rep, graph_rep,
            graph_sizes, put_indices) # (sum(m_i), d)

        if self.args.placeholder_input:
            scores = (layer_target_rep * target_rep).sum(dim=-1)
        else:
            scores = self.mlp(layer_target_rep) # (n_targets * n_candidates)
        scores = scores.reshape(n_targets) # (n_targets, n_candidates)

        return scores


