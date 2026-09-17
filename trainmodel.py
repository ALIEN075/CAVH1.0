import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import time
import numpy as np

from loss import VegHeightLoss, height_metrics, merge_metrics, format_metrics


def train_model(model, train_dataloader, val_dataloader, modelpath, num_epochs=200,
                learning_rate=1e-3, device='cuda', optimizer_type='AdamW',
                criterion=None, fit_lds=True):
    model.to(device)

    if criterion is None:
        criterion = VegHeightLoss(
            low_min=0.1, forest_thr=3.0,
            delta_nonveg=0.5, delta_low=1.0, delta_forest=3.0,
            w_nonveg=1.0, w_low=1.0, w_forest=1.5,
            use_lds=True, lds_hmin=3.0, lds_hmax=50.0, lds_bins=47,
            lds_alpha=0.5, lds_wmax=8.0,
        )
    criterion = criterion.to(device)

    # Estimate forest height histogram once before training for stable LDS weights
    if fit_lds and hasattr(criterion, 'fit_lds'):
        criterion.fit_lds(train_dataloader)

    params = model.parameters()
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=1e-3)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    train_losses = []
    val_losses = []
    val_rmses = []

    best_val_loss = float('inf')
    best_val_rmse = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        train_running_loss = 0.0
        train_total_valid_pixels = 0
        train_batches = 0
        train_terms = {}

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            valid_pixels = torch.sum(labels != 0).item()
            train_total_valid_pixels += valid_pixels
            if valid_pixels == 0:
                continue

            optimizer.zero_grad()
            height_pred = model(inputs)
            reg_loss = criterion(height_pred, labels)
            reg_loss.backward()
            optimizer.step()

            train_running_loss += reg_loss.item()
            train_batches += 1
            for k, v in criterion.last_terms.items():
                train_terms[k] = train_terms.get(k, 0.0) + v

        # ---- validate ----
        model.eval()
        val_running_loss = 0.0
        val_total_valid_pixels = 0
        val_batches = 0
        val_metrics = {}

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                valid_pixels = torch.sum(labels != 0).item()
                val_total_valid_pixels += valid_pixels
                if valid_pixels == 0:
                    continue

                height_pred = model(inputs)
                reg_loss = criterion(height_pred, labels)

                val_running_loss += reg_loss.item()
                val_batches += 1
                merge_metrics(val_metrics, height_metrics(height_pred, labels))

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if train_batches > 0:
            epoch_train_loss = train_running_loss / train_batches
            train_losses.append(epoch_train_loss)
        else:
            epoch_train_loss = float('nan')

        if val_batches > 0:
            epoch_val_loss = val_running_loss / val_batches
            val_losses.append(epoch_val_loss)

            epoch_val_rmse = np.sqrt(val_metrics['all'][0] / max(val_metrics['all'][1], 1)) \
                if 'all' in val_metrics else float('nan')
            val_rmses.append(epoch_val_rmse)

            term_str = ", ".join(
                f"{k}: {v / max(train_batches, 1):.4f}" for k, v in train_terms.items())
            print(f'Epoch [{epoch+1}/{num_epochs}], LR: {current_lr:.6f}')
            print(f'Train - Total Loss: {epoch_train_loss:.4f} | {term_str} | '
                  f'Valid Pixels: {train_total_valid_pixels}')
            print(f'Val   - Total Loss: {epoch_val_loss:.4f} | {format_metrics(val_metrics)} | '
                  f'Valid Pixels: {val_total_valid_pixels}')

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_val_rmse = epoch_val_rmse
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                    'val_rmse': epoch_val_rmse,
                    'best_val_loss': best_val_loss,
                }, modelpath)
                print(f"Best model saved at epoch {epoch+1} with validation loss "
                      f"{best_val_loss:.4f} (RMSE {best_val_rmse:.3f} m)")

    end_time = time.time()
    print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes!")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_rmses': val_rmses,
    }