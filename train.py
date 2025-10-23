# ==========================
# ⚙️ Setup + Dataset
# ==========================
import os, random, torch, datetime, time, zipfile
from io import BytesIO
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from dataset import LightLabDataset, split_dataset
from file_paths import patterns

# --------------------------
# Logging
# --------------------------
def log_message(msg):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_path, "a") as f:
        f.write(f"{timestamp} {msg}\n")
    print(msg)

# --------------------------
# Paths
# --------------------------
data_root = "./data"
results_root = os.path.join(os.path.dirname(data_root), "PredictFromOff256")
os.makedirs(results_root, exist_ok=True)
log_path = os.path.join(results_root, "training_log.txt")

# --------------------------
# Data split
# --------------------------
ds = LightLabDataset(patterns, frame_range=(1001, 1458), strict=True)

train_ds, val_ds = split_dataset(ds, val_fraction=0.01)
train_dl = DataLoader(train_ds, batch_size=18, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=18)


log_message(f"Train images: {len(train_ds)}, Validation images: {len(val_ds)}")


# ==========================
#  UNet 7-channel
# ==========================
class UNet7(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.enc1 = conv_block(8, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.bottom = conv_block(512, 1024)
        self.dec4 = conv_block(1024 + 512, 512)
        self.dec3 = conv_block(512 + 256, 256)
        self.dec2 = conv_block(256 + 128, 128)
        self.dec1 = conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, 3, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottom(self.pool(e4))
        d4 = F.interpolate(b, scale_factor=2)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = F.interpolate(d4, scale_factor=2)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, scale_factor=2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)

# ==========================
#  Training Settings
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet7().to(device)
criterion_L1 = nn.L1Loss()
criterion_L2 = nn.MSELoss()

base_lr = 1e-4
fine_tune_lr = 5e-5
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
scaler = torch.amp.GradScaler('cuda')

warmup_epochs = 60
fine_tune_epochs = 60
final_l1_epochs = 30
total_epochs = warmup_epochs + fine_tune_epochs + final_l1_epochs

# Load latest checkpoint if available
ckpt_files = [f for f in os.listdir(results_root) if f.endswith(".pt")]
if ckpt_files:
    latest_ckpt = sorted(ckpt_files)[-1]
    ckpt_path = os.path.join(results_root, latest_ckpt)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    start_epoch = int(latest_ckpt.split("_")[-1].split(".")[0])
    log_message(f"🔄 Loaded checkpoint: {ckpt_path}")
else:
    start_epoch = 0

# ==========================
# Gradient loss (safe)
# ==========================
def gradient_loss(pred, target):
    gx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    gy_target = target[:, :, 1:, :] - target[:, :, :-1, :]

    min_h = min(gx_pred.shape[2], gy_pred.shape[2], gx_target.shape[2], gy_target.shape[2])
    min_w = min(gx_pred.shape[3], gy_pred.shape[3], gx_target.shape[3], gy_target.shape[3])

    gx_pred = gx_pred[:, :, :min_h, :min_w]
    gy_pred = gy_pred[:, :, :min_h, :min_w]
    gx_target = gx_target[:, :, :min_h, :min_w]
    gy_target = gy_target[:, :, :min_h, :min_w]

    return ((gx_pred - gx_target)**2 + (gy_pred - gy_target)**2).mean()

# ==========================
# Training Loop
# ==========================
for epoch in range(start_epoch, total_epochs):
    epoch_start = time.time()

    if epoch < warmup_epochs:
        loss_fn = criterion_L1
        current_lr = base_lr
        use_grad = False
    elif epoch < warmup_epochs + fine_tune_epochs:
        loss_fn = criterion_L2
        current_lr = fine_tune_lr
        use_grad = True
    else:
        loss_fn = criterion_L1
        current_lr = fine_tune_lr
        use_grad = False

    for g in optimizer.param_groups:
        g['lr'] = current_lr

    model.train()
    running_loss = 0.0
    for img_in, img_off, _ in train_dl:
        img_in, img_off = img_in.to(device), img_off.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            pred = model(img_in)
            loss = loss_fn(pred, img_off)
            if use_grad:
                loss += 0.2 * gradient_loss(pred, img_off)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_dl)

    if (epoch + 1) % 5 == 0:
        elapsed = time.time() - epoch_start
        remaining = (total_epochs - epoch - 1) * elapsed
        log_message(f"Epoch {epoch+1}/{total_epochs} | LR: {current_lr:.5f} | Train Loss: {avg_loss:.6f} | Estimated remaining time: {remaining/60:.2f} min")

        save_path = os.path.join(results_root, f"model_epoch_{epoch+1:02d}.pt")
        torch.save(model.state_dict(), save_path)
        log_message(f"💾 Checkpoint saved for epoch {epoch+1}")

        epoch_folder = os.path.join(results_root, f"epoch_{epoch+1:02d}")
        os.makedirs(epoch_folder, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for img_in, img_off, filenames in val_dl:
                img_in = img_in.to(device)
                pred = model(img_in).cpu()
                for j in range(pred.size(0)):
                    pred_img = transforms.ToPILImage()(pred[j].clamp(0,1))
                    out_path = os.path.join(epoch_folder, filenames[j]+".jpeg")
                    print(out_path)
                    pred_img.save(out_path)
        log_message(f"Validation predictions saved to: {epoch_folder}")

# ==========================
# Final model save
# ==========================
final_pt_path = os.path.join(results_root, "light_only_predictor_256_final.pt")
torch.save(model.state_dict(), final_pt_path)
log_message(f"✅ Final model weights saved to: {final_pt_path}")

# ============================================================
# 🚀 EXPORT NUKE-COMPATIBLE TORCHSCRIPT MODEL
# ============================================================
export_model = UNet7().to(device)
state_dict = torch.load(final_pt_path, map_location=device)
export_model.load_state_dict(state_dict)
export_model.eval()

dummy_input = torch.randn(1, 8, 256, 256).to(device)
traced_model = torch.jit.trace(export_model, dummy_input)
export_path = os.path.join(results_root, "light_only_predictor_256_final_traced.pt")
traced_model.save(export_path)

print(f"✅ Traced model saved for Nuke at:\n{export_path}")

