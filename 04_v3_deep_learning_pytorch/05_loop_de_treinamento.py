"""
Fragmento de estudo (V3): loop de treino em PyTorch.

Mostra a estrutura minima de treino/validacao com early stopping.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def treinar_epocas(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    epochs=50,
    patience=7,
):
    """
    ETAPA 1 - Treino em mini-batches.
    ETAPA 2 - Validacao por AUC-ROC.
    ETAPA 3 - Early stopping.
    """
    best_val_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ETAPA 1 - Treino.
        model.train()
        for batch_cat, batch_num, batch_y in train_loader:
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(batch_cat, batch_num)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # ETAPA 2 - Validacao.
        model.eval()
        val_probs = []
        val_true = []
        with torch.no_grad():
            for batch_cat, batch_num, batch_y in val_loader:
                logits = model(batch_cat.to(device), batch_num.to(device))
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_true.extend(batch_y.numpy())

        val_auc = roc_auc_score(np.array(val_true), np.array(val_probs))
        print(f"Epoca {epoch + 1:02d}/{epochs} - Val AUC: {val_auc:.4f}")

        # ETAPA 3 - Early stopping.
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping ativado na epoca {epoch + 1}.")
                break

    return best_val_auc


if __name__ == "__main__":
    print("Use treinar_epocas(...) dentro do script final da V3.")
