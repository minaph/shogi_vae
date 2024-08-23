import torch
import torch.nn as nn
import torch.nn.functional as F


class ShogiVAE(nn.Module):
    def __init__(self, input_channels=104, latent_dim=128):
        super(ShogiVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 3 * 3)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # ここを0に変更
            nn.ReLU(),
            nn.Conv2d(64, input_channels, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 3, 3)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


if __name__ == "__main__":
    # モデルの初期化
    model = ShogiVAE(input_channels=104)

    # ダミー入力データの作成
    dummy_input = torch.randn(1, 104, 9, 9)

    # モデルの順伝播
    recon_batch, mu, logvar = model(dummy_input)

    # 出力サイズの確認
    print(f"\n実際の出力サイズ:")
    print(f"Input size: {dummy_input.shape}")
    print(f"Reconstructed output size: {recon_batch.shape}")
    print(f"Mu size: {mu.shape}")
    print(f"Logvar size: {logvar.shape}")
