import torch
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
import cshogi
from cshogi import PIECE_TYPES, MAX_PIECES_IN_HAND  

# FEATURES_NUM is defined as len(PIECE_TYPES) * 2 + sum(MAX_PIECES_IN_HAND) * 2.

FEATURES_NUM = len(PIECE_TYPES) * 2 + sum(MAX_PIECES_IN_HAND) * 2

class SfenDataset(Dataset):
    def __init__(self, file_path):
        """
        SFENデータセットを初期化する

        Parameters:
        file_path (str): SFENデータを含むテキストファイルのパス
        """
        self.sfen_data = []
        with open(file_path, "r") as f:
            for line in f:
                self.sfen_data.append(line.strip())

    def __len__(self):
        return len(self.sfen_data)

    def __getitem__(self, idx):
        sfen = self.sfen_data[idx]
        return self.preprocess_sfen_data(sfen)

    @staticmethod
    def preprocess_sfen_data(sfen_data):
        """
        SFENデータを(FEATURES_NUM, 9, 9)のテンソルに変換する前処理関数

        Parameters:
        sfen_data (str): SFEN形式の局面データ

        Returns:
        torch.Tensor: (FEATURES_NUM, 9, 9)の形状を持つPyTorchテンソル
        """
        board = cshogi.Board()
        board.set_position(sfen_data)
        features = np.zeros((FEATURES_NUM, 9, 9), dtype=np.float32)
        board.piece_planes(features)
        return torch.from_numpy(features)


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[Dataset, Dataset, Dataset]:
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
    ), "Ratios must sum to 1"

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  # シード値を設定して再現性を確保
    )

    return train_dataset, val_dataset, test_dataset


# 使用例
if __name__ == "__main__":
    # SFENデータセットの作成
    dataset = SfenDataset("records2016_10818.sfen.txt")

    # DataLoaderの作成
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # データの読み込みとバッチの表示
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        print(f"Batch type: {batch.dtype}")
        break  # 最初のバッチだけを表示

    # データセットの大きさを表示
    print(f"Dataset size: {len(dataset)}")

    # 単一のサンプルを取得して表示
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample type: {sample.dtype}")
    print(f"Non-zero elements in sample: {torch.count_nonzero(sample)}")
