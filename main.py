import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# モデルの定義（SimpleMLP）
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)  # input: 784 -> output: 128
        self.fc2 = nn.Linear(128, 10)   # input: 128 -> output: 10（10クラス分類）

    def forward(self, x):
        # バッチサイズに合わせてリシェイプ。x の形状は (batch, 1, 28, 28)
        x_reshaped = x.view(x.shape[0], -1)  # (batch, 784)
        h = self.fc1(x_reshaped)
        z = torch.sigmoid(h)
        y_hat = self.fc2(z)
        return y_hat

# デバイス設定（GPUがあればGPU、なければCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 作成したモデルをデバイスへ転送
model = SimpleMLP().to(device)
print(model)

# --- モデルのロード ---
loaded_model = SimpleMLP().to(device)

# FutureWarning の対策として weights_only=True を指定（※環境によっては不要）
loaded_model.load_state_dict(torch.load("modelwithBatch.pth", map_location=device, weights_only=True))
loaded_model.eval()  # 推論モードに切り替え

print("モデルをロードしました。")

# --- DataLoader の作成（ここでは既に train_subset, val_subset がある前提） ---
batch_size = 64
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# --- 検証データから1枚のテスト画像を取得して予測 ---
with torch.no_grad():
    # val_loader から最初のバッチを取得
    val_batch = next(iter(val_loader))
    images, labels = val_batch  # images: [batch, 1, 28, 28], labels: [batch]

    # バッチの先頭の画像をテスト画像として選択
    test_image = images[0].unsqueeze(0).to(device)  # 形状: [1, 1, 28, 28]
    true_label = labels[0].item()

    # モデルに入力して予測
    output = loaded_model(test_image)  # 出力形状: [1, 10]
    probabilities = torch.softmax(output, dim=1)  # 各クラスの予測確率
    _, predicted_class = torch.max(probabilities, 1)

print("正解ラベル:", true_label)
print("予測されたクラス:", predicted_class.item())
print("各クラスの確率:", probabilities.cpu().numpy())

# --- テスト画像の表示 ---
# test_image の形状は [1, 1, 28, 28] なので、squeeze して [28, 28] に変換
test_image_disp = test_image.squeeze().cpu().numpy()
plt.imshow(test_image_disp, cmap='gray')
plt.title(f"True Label: {true_label}, Predicted: {predicted_class.item()}")
plt.axis('off')
plt.show()
