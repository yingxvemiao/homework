
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from scipy.stats import spearmanr, kendalltau
# 1
import os, random

# 2
def _seed_all(seed: int = 42):
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    except Exception:
        pass

warnings.filterwarnings("ignore")

# ======================= USER PATH =======================
DATA_DIR = (Path(__file__).resolve().parent / "clean")
# ========================================================

# ----------------------------- Hyperparams ----------------------------
SEED         = 42
EPOCHS_1     = 80
EPOCHS_2     = 40
BATCH_SIZE   = 64
LR1          = 1e-3
LR2          = 5e-4
WEIGHT_DECAY = 1e-4
CORR_LAMBDA  = 0.2

# ----------------------------- Utilities -----------------------------
def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    obj_cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(3, int(0.3 * len(s))):
            out[c] = s
        else:
            obj_cols.append(c)
    num_df = pd.DataFrame(out, index=df.index)
    cat_df = pd.DataFrame(index=df.index)
    for c in obj_cols:
        nunique = df[c].nunique(dropna=False)
        if nunique <= 50:
            cat_df = pd.concat([cat_df, pd.get_dummies(df[c], prefix=c, dummy_na=True)], axis=1)
    if len(cat_df.columns) > 0:
        num_df = num_df.join(cat_df, how="left").fillna(0.0)
    return num_df

def safe_spearman(y_true, y_pred):
    try:
        val = spearmanr(y_true, y_pred, nan_policy="omit").correlation
        return float(val) if val is not None else float("nan")
    except Exception:
        return float("nan")

def safe_kendall(y_true, y_pred):
    try:
        val = kendalltau(y_true, y_pred, nan_policy="omit").correlation
        return float(val) if val is not None else float("nan")
    except Exception:
        return float("nan")

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return dict(MSE=np.nan, RMSE=np.nan, MAE=np.nan, MAPE=np.nan, MedianAE=np.nan,
                    R2=np.nan, nRMSE=np.nan, Spearman=np.nan, Kendall=np.nan)
    yt = y_true[mask]; yp = y_pred[mask]
    mse = mean_squared_error(yt, yp)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(yt, yp)
    mape = float(np.mean(np.abs(yt - yp) / np.maximum(np.abs(yt), 1e-6)) * 100.0)
    medae = float(np.median(np.abs(yt - yp)))
    r2 = r2_score(yt, yp)
    rng = float(np.max(yt) - np.min(yt)) if len(yt) > 0 else float("nan")
    nrmse = float(rmse / rng) if rng and rng > 0 else float("nan")
    rho = safe_spearman(yt, yp)
    tau = safe_kendall(yt, yp)
    return dict(MSE=float(mse), RMSE=rmse, MAE=float(mae), MAPE=mape,
                MedianAE=medae, R2=float(r2), nRMSE=nrmse, Spearman=rho, Kendall=tau)

# >>> ADDED: differentiable Pearson correlation for correlation-aware fine-tuning
def pearson_corr_torch(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a.reshape(-1); b = b.reshape(-1)
    a = a - a.mean(); b = b - b.mean()
    num = (a * b).sum()
    den = torch.sqrt((a * a).sum() + eps) * torch.sqrt((b * b).sum() + eps)
    return num / (den + eps)

# ----------------------------- Model -----------------------------
class Norm1d(nn.Module):
    def __init__(self, d, kind="batch"):
        super().__init__()
        if kind == "batch":
            self.norm = nn.BatchNorm1d(d)
        elif kind == "layer":
            self.norm = nn.LayerNorm(d)
        else:
            self.norm = nn.Identity()
    def forward(self, x):
        return self.norm(x)

class Block(nn.Module):
    def __init__(self, in_dim, out_dim, pdrop=0.1, norm_kind="batch"):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = Norm1d(out_dim, kind=norm_kind)
        self.do  = nn.Dropout(pdrop)
    def forward(self, x):
        x = self.lin(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.do(x)
        return x

class DeepRankMLP(nn.Module):
    def __init__(self, in_dim, norm_kind="batch"):
        super().__init__()
        h = 256
        self.net = nn.Sequential(
            Block(in_dim, h, pdrop=0.1, norm_kind=norm_kind),
            Block(h, h, pdrop=0.1, norm_kind=norm_kind),
            Block(h, h//2, pdrop=0.1, norm_kind=norm_kind),
            nn.Linear(h//2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ----------------------------- Train One -----------------------------
def train_one(X_train, y_train_norm, X_val, y_val_norm, X_test, y_test_norm,
              y_min, y_max, device="cpu"):
    torch.manual_seed(SEED); np.random.seed(SEED)

    norm_kind = "batch" if len(X_train) >= 2 else "layer"
    model = DeepRankMLP(X_train.shape[1], norm_kind=norm_kind).to(device)
    huber = nn.SmoothL1Loss(beta=1.0)

    tr_dl = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                     torch.tensor(y_train_norm, dtype=torch.float32)),
                       batch_size=BATCH_SIZE, shuffle=True)
    # 8
    try:
        _eff_bs = max(2, min(BATCH_SIZE, len(tr_dl.dataset)))
        _drop_last = (len(tr_dl.dataset) % _eff_bs == 1)
        tr_dl = DataLoader(tr_dl.dataset, batch_size=_eff_bs, shuffle=True, drop_last=_drop_last)
    except Exception:
        pass

    va_dl = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                     torch.tensor(y_val_norm, dtype=torch.float32)),
                       batch_size=256, shuffle=False)

    def denorm(y_norm):
        return y_norm * max(y_max - y_min, 1e-9) + y_min

    # Stage 1 with early stopping on val MSE (rank units)
    opt = torch.optim.AdamW(model.parameters(), lr=LR1, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)
    best_state = None
    best_val_mse = float("inf")
    patience = 12
    bad = 0

    for ep in range(EPOCHS_1):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = huber(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # validate
        model.eval()
        with torch.no_grad():
            yp, yt = [], []
            for xb, yb in va_dl:
                xb = xb.to(device)
                yp.append(model(xb).cpu().numpy())
                yt.append(yb.cpu().numpy())
        if len(yp) > 0:
            y_pred_rank = denorm(np.concatenate(yp).reshape(-1))
            y_true_rank = denorm(np.concatenate(yt).reshape(-1))
            v_mse = mean_squared_error(y_true_rank, y_pred_rank)
            if v_mse < best_val_mse - 1e-8:
                best_val_mse = v_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Stage 2: correlation-aware fine-tune with early stopping
    opt2 = torch.optim.AdamW(model.parameters(), lr=LR2, weight_decay=WEIGHT_DECAY)
    best_state2 = None
    best_val_mse2 = float("inf")
    patience2 = 8
    bad2 = 0

    for ep in range(EPOCHS_2):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(device); yb = yb.to(device)
            opt2.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = huber(pred, yb)
            loss_corr = 1.0 - pearson_corr_torch(pred, yb)
            loss_total = loss + CORR_LAMBDA * loss_corr
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()

        # val
        model.eval()
        with torch.no_grad():
            yp, yt = [], []
            for xb, yb in va_dl:
                xb = xb.to(device)
                yp.append(model(xb).cpu().numpy())
                yt.append(yb.cpu().numpy())
        if len(yp) > 0:
            y_pred_rank = denorm(np.concatenate(yp).reshape(-1))
            y_true_rank = denorm(np.concatenate(yt).reshape(-1))
            v_mse = mean_squared_error(y_true_rank, y_pred_rank)
            if v_mse < best_val_mse2 - 1e-8:
                best_val_mse2 = v_mse
                best_state2 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad2 = 0
            else:
                bad2 += 1
                if bad2 >= patience2:
                    break

    if best_state2 is not None:
        model.load_state_dict(best_state2)

    # prediction helpers
    def predict_rank(X_np: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_np).float().to(device)
            yhat_norm = model(X_t).cpu().numpy().reshape(-1)
        return denorm(yhat_norm)

    # attach simple dropout test-time averaging without touching predict_rank
    # 3
    def predict_rank_tta(X_np: np.ndarray, n_pass: int = 5) -> np.ndarray:
        outs = []
        with torch.no_grad():
            X_t = torch.from_numpy(X_np).float().to(device)
            for _ in range(max(1, n_pass)):
                model.train()  # enable dropout
                outs.append(model(X_t).cpu().numpy().reshape(-1))
            model.eval()
        yhat_norm = np.mean(np.vstack(outs), axis=0)
        return denorm(yhat_norm)

    # add method to model for external use if需要
    model.predict_rank_tta = predict_rank_tta

    return model, predict_rank

# ----------------------------- Main -----------------------------
def main():
    # 4
    base_seed = SEED
    _seed_all(base_seed)

    assert DATA_DIR.exists() and DATA_DIR.is_dir(), f"数据目录不存在: {DATA_DIR}"
    csvs = sorted(list(DATA_DIR.rglob('*.csv')))
    assert len(csvs) > 0, f"在目录 {DATA_DIR} 下未找到任何CSV文件"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for i, csv_path in enumerate(csvs, start=1):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception:
            df = pd.read_csv(csv_path, encoding="gb18030")

        label_col = df.columns[0]
        print(f"\\n=== [{i}/{len(csvs)}] 学科: {csv_path.name} | 排名列: {label_col} ===")

        df = df.reset_index(drop=False).rename(columns={"index":"row_id"})
        y_raw = pd.to_numeric(df[label_col], errors="coerce").astype(float)
        X = df.drop(columns=[label_col])
        X = to_numeric_df(X)
        mask = y_raw.notna()
        X = X.loc[mask].copy()
        y_raw = y_raw.loc[mask].astype(float)
        row_ids = df.loc[mask, "row_id"].to_numpy()

        X_np = X.to_numpy(dtype=float)
        y_np = y_raw.to_numpy(dtype=float)

        # split
        if len(X_np) >= 8:
            X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
                X_np, y_np, row_ids, test_size=0.2, random_state=base_seed)
        else:
            X_train, X_test, y_train, y_test, idx_tr, idx_te = X_np, X_np, y_np, y_np, row_ids, row_ids

        # normalize y to [0,1] by train min max
        y_min, y_max = float(np.min(y_train)), float(np.max(y_train))
        rng = max(y_max - y_min, 1e-9)
        y_trn = (y_train - y_min) / rng
        y_ten = (y_test  - y_min) / rng

        # split train/val
        if len(X_train) >= 6:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_train, y_trn, test_size=0.25, random_state=base_seed)
        else:
            X_tr, X_va, y_tr, y_va = X_train, X_train, y_trn, y_trn

        # feature scaling
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_test)

        # train
        model, predict_rank = train_one(
            X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_ten, y_min=y_min, y_max=y_max, device=device
        )

        # predict base and TTA
        y_pred_te = predict_rank(X_te_s)
        try:
            y_pred_te_tta = model.predict_rank_tta(X_te_s, n_pass=5)
            y_pred_te = 0.7 * y_pred_te + 0.3 * y_pred_te_tta
        except Exception:
            pass

        # validation-based linear calibration to reduce bias
        # 5
        try:
            y_pred_va = predict_rank(X_va_s)
            y_true_va = y_va * (y_max - y_min) + y_min
            if len(y_pred_va) == len(y_true_va) and len(y_true_va) >= 5:
                a, b = np.polyfit(y_pred_va, y_true_va, 1)
                if np.isfinite(a) and np.isfinite(b):
                    a = float(np.clip(a, 0.5, 2.0))
                    b = float(np.clip(b, -2.0 * abs(np.mean(y_true_va)), 2.0 * abs(np.mean(y_true_va))))
                    y_pred_te = a * y_pred_te + b
        except Exception:
            pass

        # 6
        try:
            y_pred_te = np.clip(y_pred_te, y_min, y_max)
        except Exception:
            pass

        y_true_te = y_np[idx_te]

        # 7
        try:
            if np.allclose(y_true_te, np.round(y_true_te)):
                y_pred_te = np.round(y_pred_te)
        except Exception:
            pass

        metrics = compute_metrics(y_true_te, y_pred_te)
        print("[测试] " + "  ".join([
            f"MSE={metrics['MSE']:.4f}",
            f"RMSE={metrics['RMSE']:.2f}",
            f"MAE={metrics['MAE']:.2f}",
            f"R2={metrics['R2']:.3f}",
            f"Spearman={metrics['Spearman']:.3f}",
            f"Kendall={metrics['Kendall']:.3f}",
        ]))

        pred_df = pd.DataFrame({
            "row_id": row_ids[idx_te],
            "true_rank": y_true_te,
            "pred_rank": y_pred_te,
        })
        safe = "".join([ch if ch.isalnum() or ch in ("-","_") else "_" for ch in csv_path.stem])
        pred_path = (Path(__file__).resolve().parent / f"predictions_{safe}.csv")
        try:
            pred_df.to_csv(pred_path, index=False)
            print(f"已保存测试对比表: {pred_path.resolve()}")
        except Exception:
            pass

        row = {"subject_csv": csv_path.name}
        row.update(metrics)
        rows.append(row)

    if rows:
        out = pd.DataFrame(rows).sort_values("MSE")
        out_path = (Path(__file__).resolve().parent / "deep_learning.csv")
        out.to_csv(out_path, index=False)
        print("\\n=== 总结（按 MSE 升序） ===")
        print(out.to_string(index=False))
        print(f"\\n结果已保存到: {out_path.resolve()}")

if __name__ == "__main__":
    main()
