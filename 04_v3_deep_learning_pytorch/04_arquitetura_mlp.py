"""
Fragmento de estudo (V3): arquitetura Tabular MLP.

Mostra a estrutura da rede com embeddings para categoricas.
"""
import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """Rede tabular com embeddings + bloco denso."""

    def __init__(
        self,
        embedding_sizes,
        num_numerical_cols,
        emb_dropout_p=0.1,
        hidden_sizes=[256, 128, 64],
        dropout_p=0.3,
    ):
        super().__init__()

        # ETAPA 1 - Camadas de embedding para categoricas.
        self.embeds = nn.ModuleList(
            [nn.Embedding(num_categories, emb_size) for num_categories, emb_size in embedding_sizes]
        )
        self.emb_dropout = nn.Dropout(emb_dropout_p)
        self.bn_num = nn.BatchNorm1d(num_numerical_cols)

        # ETAPA 2 - Bloco MLP.
        num_emb_cols = sum([emb_size for _, emb_size in embedding_sizes])
        input_size = num_emb_cols + num_numerical_cols
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Mish())
            layers.append(nn.Dropout(dropout_p))
            input_size = hidden_size

        # ETAPA 3 - Saida em logit.
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Flatten())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        """ETAPA 4 - Fluxo: embeddings + numericas -> MLP -> logit."""
        if x_cat is None and x_num is None:
            raise ValueError("x_cat e x_num nao podem ser ambos None.")

        # Caminho categorico.
        if len(self.embeds) > 0 and x_cat is not None:
            emb_outputs = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeds)]
            x_cat_emb = self.emb_dropout(torch.cat(emb_outputs, dim=1))
            batch_size = x_cat.size(0)
            device = x_cat.device
        else:
            batch_size = x_num.size(0)
            device = x_num.device
            x_cat_emb = torch.empty(batch_size, 0, device=device)

        # Caminho numerico.
        if x_num is not None and x_num.shape[1] > 0:
            x_num_out = self.bn_num(x_num)
        else:
            x_num_out = torch.empty(batch_size, 0, device=device)

        # Fusao e inferencia.
        x_combined = torch.cat([x_cat_emb, x_num_out], dim=1)
        return self.mlp(x_combined)


if __name__ == "__main__":
    print("TabularMLP pronto para ser importado no treino V3.")
