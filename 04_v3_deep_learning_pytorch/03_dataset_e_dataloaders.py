"""
Fragmento de estudo (V3): Dataset e DataLoader.

Mostra como preparar batches para treino em PyTorch.
"""
import torch
from torch.utils.data import DataLoader, Dataset


class LeadsDataset(Dataset):
    """ETAPA 1 - Encapsular arrays em um Dataset PyTorch."""

    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


def criar_dataloader(dataset, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True):
    """ETAPA 2 - Criar DataLoader para ler em mini-batches."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


if __name__ == "__main__":
    print("[*] Dataset/DataLoader prontos para uso no treino da V3.")
