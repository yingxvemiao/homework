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

warnings.filterwarnings("ignore")

# ======================= USER PATH =======================
DATA_DIR = (Path(__file__).resolve().parent / "clean")   # 修改为你的CSV目录
# ========================================================

# ----------------------------- Hyperparams ----------------------------
SEED         = 42
EPOCHS_1     = 80
EPOCHS_2     = 40
BATCH_SIZE   = 64
LR1          = 3e-3
LR2          = 1e-3
WEIGHT_DECAY = 1e-4

# >>> ADDED: ensemble & correlation hyperparams
ENSEMBLE_MODELS = 5
SEED_STEP       = 11
KFOLD_MODELS    = 5
EPOCHS_3        = 20
LR3             = 5e-4
CORR_LAMBDA     = 0.2

# ----------------------------- Metrics -----------------------------

def mse_np(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def mape_np(y_true, y_pred, eps=1e-6):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def medae_np(y_true, y_pred):
    return float(np.median(np.abs(y_true - y_pred)))

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
    mape = mape_np(yt, yp)
    medae = medae_np(yt, yp)
    try:
        r2 = r2_score(yt, yp)
    except Exception:
        r2 = float("nan")
    rng = float(np.max(yt) - np.min(yt)) if yt.size > 0 else float("nan")
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
# ----------------------------- Features -----------------------------

def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Numericize as much as possible; one-hot low-cardinality object columns."""
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
    return num_df if cat_df.shape[1] == 0 else pd.concat([num_df, cat_df], axis=1)

def split_60_20_20_idx(idx, seed=42):
    """Split only indices (to keep original row ids)."""
    idx = np.array(idx)
    idx_train, idx_tmp = train_test_split(idx, train_size=0.6, random_state=seed, shuffle=True)
    idx_val, idx_test   = train_test_split(idx_tmp, train_size=0.5, random_state=seed, shuffle=True)
    return idx_train, idx_val, idx_test

# ----------------------------- Model -----------------------------

class Norm1d(nn.Module):
    def __init__(self, dim, kind="batch"):
        super().__init__()
        if kind == "batch":
            self.norm = nn.BatchNorm1d(dim)
        elif kind == "layer":
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()
    def forward(self, x):
        return self.norm(x)

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, pdrop=0.25, norm_kind="batch"):
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
        h1 = max(64, 2 * in_dim)
        h2 = max(64, in_dim)
        self.blocks = nn.ModuleList([
            MLPBlock(in_dim, h1,   pdrop=0.25, norm_kind=norm_kind),
            MLPBlock(h1,    h2,   pdrop=0.25, norm_kind=norm_kind),
            MLPBlock(h2,    128,  pdrop=0.25, norm_kind=norm_kind),
            MLPBlock(128,   64,   pdrop=0.25, norm_kind=norm_kind),
            MLPBlock(64,    32,   pdrop=0.25, norm_kind=norm_kind),
        ])
        self.out = nn.Linear(32, 1)
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.out(x).squeeze(1)

def freeze_first_half(model: DeepRankMLP, freeze: bool = True):
    """Freeze parameters of the first half of blocks for fine-tuning stage."""
    k = len(model.blocks) // 2
    for i, b in enumerate(model.blocks):
        for p in b.parameters():
            p.requires_grad = (not freeze) or (i >= k)

# ----------------------------- Training -----------------------------

def train_one(X_train, y_train_norm, X_val, y_val_norm, X_test, y_test_norm,
              y_min, y_max, device="cpu"):
    """
    Train on normalized targets y_norm in [0,1].
    Keep best model by validation MSE measured in original rank units.
    Return model and a function to predict (denormalized) ranks.
    """
    torch.manual_seed(SEED); np.random.seed(SEED)

    # Choose norm kind: fallback to LayerNorm when train set too small (<2)
    norm_kind = "batch" if len(X_train) >= 2 else "layer"
    model = DeepRankMLP(X_train.shape[1], norm_kind=norm_kind).to(device)
    huber = nn.SmoothL1Loss(beta=1.0)  # robust to outliers

    # Stage 1: full training
    opt = torch.optim.AdamW(model.parameters(), lr=LR1, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)

    tr_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train_norm).float())
    va_ds = TensorDataset(torch.from_numpy(X_val).float(),   torch.from_numpy(y_val_norm).float())

    # Avoid 1-sample batch in training
    bs_eff = min(BATCH_SIZE, len(tr_ds))
    drop_last = (len(tr_ds) >= 2)
    tr_dl = DataLoader(tr_ds, batch_size=bs_eff, shuffle=True, drop_last=drop_last)
    va_dl = DataLoader(va_ds, batch_size=min(BATCH_SIZE, len(va_ds)), shuffle=False)

    def denorm(y_hat_norm: np.ndarray) -> np.ndarray:
        return y_min + np.clip(y_hat_norm, 0.0, 1.0) * (y_max - y_min)

    best_state = None
    best_val_mse_rank = float("inf")

    def run_epochs(epochs):
        nonlocal best_state, best_val_mse_rank
        for ep in range(1, epochs + 1):
            if len(tr_dl) == 0:
                break
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = huber(pred, yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()

            # validation (measure MSE in rank units)
            model.eval()
            with torch.no_grad():
                yp, yt = [], []
                for xb, yb in va_dl:
                    xb = xb.to(device)
                    yp.append(model(xb).cpu().numpy())
                    yt.append(yb.cpu().numpy())
                if len(yp) == 0:
                    continue
                y_pred_norm = np.concatenate(yp).reshape(-1)
                y_true_norm = np.concatenate(yt).reshape(-1)
                y_pred_rank = denorm(y_pred_norm)
                y_true_rank = denorm(y_true_norm)
                v_mse_rank = mse_np(y_true_rank, y_pred_rank)
                if v_mse_rank < best_val_mse_rank:
                    best_val_mse_rank = v_mse_rank
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    run_epochs(EPOCHS_1)

    # Stage 2: freeze and fine-tune
    freeze_first_half(model, True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR2, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    run_epochs(EPOCHS_2)

    # >>> ADDED: Stage 3 - correlation-aware fine-tuning to promote rank consistency
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR3, weight_decay=WEIGHT_DECAY)

    def run_epochs_corr(epochs):
        nonlocal best_state, best_val_mse_rank
        for ep in range(1, epochs + 1):
            if len(tr_dl) == 0:
                break
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss_rank = huber(pred, yb)
                corr = pearson_corr_torch(pred, yb)
                loss = loss_rank + CORR_LAMBDA * (1.0 - corr)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            # validate with rank-space MSE like Stage 1/2
            model.eval()
            with torch.no_grad():
                yp, yt = [], []
                for xb, yb in va_dl:
                    xb = xb.to(device)
                    yp.append(model(xb).cpu().numpy())
                    yt.append(yb.cpu().numpy())
                if len(yp) == 0:
                    continue
                y_pred_norm = np.concatenate(yp).reshape(-1)
                y_true_norm = np.concatenate(yt).reshape(-1)
                y_pred_rank = denorm(y_pred_norm)
                y_true_rank = denorm(y_true_norm)
                v_mse_rank = mse_np(y_true_rank, y_pred_rank)
                if v_mse_rank < best_val_mse_rank:
                    best_val_mse_rank = v_mse_rank
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    run_epochs_corr(EPOCHS_3)


    if best_state is not None:
        model.load_state_dict(best_state)

    def predict_rank(X_np: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_np).float().to(device)
            yhat_norm = model(X_t).cpu().numpy().reshape(-1)
        return denorm(yhat_norm)

    return model, predict_rank

# ----------------------------- Main -----------------------------

def main():
    assert DATA_DIR.exists() and DATA_DIR.is_dir(), f"数据目录不存在: {DATA_DIR}"
    csvs = sorted(list(DATA_DIR.rglob('*.csv')))
    assert len(csvs) > 0, f"在目录 {DATA_DIR} 下未找到任何CSV文件"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for i, csv_path in enumerate(csvs, 1):
        # Read CSV robustly
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = pd.read_csv(csv_path, encoding="gb18030")

        # First column is rank
        label_col = df.columns[0]
        print(f"\n=== [{i}/{len(csvs)}] 学科: {csv_path.name} | 排名列: {label_col} ===")

        # Keep original indices for reporting
        df = df.reset_index(drop=False).rename(columns={"index":"row_id"})

        # Build X/y
        y_raw = pd.to_numeric(df[label_col], errors="coerce").astype(float)
        X = df.drop(columns=[label_col])
        X = to_numeric_df(X)

        # Valid mask on y
        mask = y_raw.notna()
        X = X.loc[mask].copy()
        y_raw = y_raw.loc[mask].astype(float)

        # Row ids for valid rows
        row_ids = df.loc[mask, "row_id"].to_numpy()

        # Impute X NaNs
        X = X.fillna(X.median(numeric_only=True))

        # Split indices 60/20/20 (so we can save test rows later)
        N = len(row_ids)
        idx_all = np.arange(N)
        idx_tr, idx_va, idx_te = split_60_20_20_idx(idx_all, seed=SEED)

        X_np = X.values
        y_np = y_raw.values

        # Feature scaling based on train only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_np[idx_tr])
        X_val   = scaler.transform(X_np[idx_va])
        X_test  = scaler.transform(X_np[idx_te])

        # Normalize y to [0,1] using all valid data bounds
        y_min = float(np.min(y_np))
        y_max_v = float(np.max(y_np))
        y_max = y_max_v if y_max_v != y_min else (y_min + 1.0)
        def norm(y): return (y - y_min) / (y_max - y_min)

        y_train = norm(y_np[idx_tr])
        y_val   = norm(y_np[idx_va])
        y_test  = norm(y_np[idx_te])

        # Train & get predictor
        _, predict_rank = train_one(
            X_train, y_train, X_val, y_val, X_test, y_test,
            y_min=y_min, y_max=y_max, device=device
        )

        # Predict on test and compute metrics
        y_pred_te = predict_rank(X_test)
        # >>> ADDED-A: multi-seed bagging
        if 'ENSEMBLE_MODELS' in globals() and ENSEMBLE_MODELS > 1:
            _seed_backup = SEED if 'SEED' in globals() else 42
            preds_list = [y_pred_te.copy()]
            for k in range(1, ENSEMBLE_MODELS):
                try:
                    SEED = _seed_backup + SEED_STEP * k
                except Exception:
                    pass
                _, _predict_k = train_one(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    y_min=y_min, y_max=y_max, device=device
                )
                preds_list.append(_predict_k(X_test))
            try:
                SEED = _seed_backup
            except Exception:
                pass
            y_pred_te = np.mean(np.vstack(preds_list), axis=0)

        # >>> ADDED-B: K-Fold on train+val for extra diversity (test set unchanged)
        try:
            from sklearn.model_selection import KFold
            idx_trva = np.concatenate([idx_tr, idx_va])
            X_trva   = X_np[idx_trva]
            y_trva   = y_np[idx_trva]
            # reuse scaler if present
            if 'scaler' in locals():
                X_trva = scaler.transform(X_trva)
            kf = KFold(n_splits=KFOLD_MODELS, shuffle=True, random_state=SEED if 'SEED' in globals() else 42)
            cv_preds = []
            for tr2, va2 in kf.split(X_trva):
                X_tr2, X_va2 = X_trva[tr2], X_trva[va2]
                y_tr2, y_va2 = y_trva[tr2], y_trva[va2]
                y_min2, y_max2 = float(np.min(y_tr2)), float(np.max(y_tr2))
                rng2 = max(y_max2 - y_min2, 1e-9)
                y_tr2n = (y_tr2 - y_min2) / rng2
                y_va2n = (y_va2 - y_min2) / rng2
                _, _predict_cv = train_one(
                    X_tr2, y_tr2n, X_va2, y_va2n, X_test, y_test,
                    y_min=y_min2, y_max=y_max2, device=device
                )
                cv_preds.append(_predict_cv(X_test))
            if len(cv_preds) > 0:
                y_pred_te = np.mean(np.vstack([y_pred_te] + cv_preds), axis=0)
        except Exception:
            pass

        # >>> ADDED: gentle clipping to observed range
        y_pred_te = np.clip(y_pred_te, y_min, y_max)

        y_true_te = y_np[idx_te]

        metrics = compute_metrics(y_true_te, y_pred_te)

        # Print compact metrics line
        print("[测试] " + "  ".join([
            f"MSE={metrics['MSE']:.4f}",
            f"RMSE={metrics['RMSE']:.2f}",
            f"MAE={metrics['MAE']:.2f}",
            f"MAPE={metrics['MAPE']:.2f}%",
            f"MedAE={metrics['MedianAE']:.2f}",
            f"R2={metrics['R2']:.3f}",
            f"nRMSE={metrics['nRMSE']:.3f}",
            f"Spearman={metrics['Spearman']:.3f}",
            f"Kendall={metrics['Kendall']:.3f}",
        ]))

        # Save per-CSV predictions table (row_id, true_rank, pred_rank)
        pred_df = pd.DataFrame({
            "row_id": row_ids[idx_te],
            "true_rank": y_true_te,
            "pred_rank": y_pred_te,
        })
        # sanitize filename
        stem = csv_path.stem
        safe = "".join([ch if ch.isalnum() or ch in ("-","_") else "_" for ch in stem])
        pred_path = (Path(__file__).resolve().parent / f"predictions_{safe}.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"已保存测试对比表: {pred_path.resolve()}")

        # Append summary row
        row = {"subject_csv": csv_path.name}
        row.update(metrics)
        rows.append(row)

    # Save summary metrics
    if rows:
        out = pd.DataFrame(rows).sort_values("MSE")
        out_path = (Path(__file__).resolve().parent / "deep_learning.csv")
        out.to_csv(out_path, index=False)
        print("\n=== 总结（按 MSE 升序） ===")
        print(out.to_string(index=False))
        print(f"\n结果已保存到: {out_path.resolve()}")

if __name__ == "__main__":
    main()
