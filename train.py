import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from preprocess import SfenDataset, split_dataset
from model import ShogiVAE, vae_loss


def train(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss / len(train_loader.dataset)


def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            val_loss += vae_loss(recon_batch, data, mu, logvar).item()

    return val_loss / len(val_loader.dataset)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データセットの準備と分割
    full_dataset = SfenDataset(args.data)
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # データローダーの準備
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # モデル、オプティマイザ、スケジューラの準備
    model = ShogiVAE(input_channels=104, latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    # トレーニングループ
    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}, Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}"
        )

        scheduler.step(val_loss)

        # モデルの保存
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), f"./epochs/shogi_vae_epoch_{epoch}.pth")

    # 学習曲線のプロット
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_curve.png")
    plt.close()

    # 最終モデルの保存
    torch.save(model.state_dict(), "shogi_vae_final.pth")

    # テストデータでの評価
    test_loss = validate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ShogiVAE")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to SFEN data file"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Save model every n epochs"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Ratio of training data"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Ratio of validation data"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="Ratio of test data"
    )

    args = parser.parse_args()
    main(args)
