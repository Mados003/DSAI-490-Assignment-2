import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).parent))

from utils.dataset import get_dataloaders
from models.ae import ConditionalAE
from models.cgan import CGANGenerator, CGANDiscriminator, NOISE_DIM
from models.lstm import ConditionalLSTM
from models.transformer import DateTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ce_loss(yl: torch.Tensor, dl: torch.Tensor,
            yt: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(yl, yt) + F.cross_entropy(dl, dt)


def train_ae(train_loader, test_loader, epochs: int, lr: float,
             device: torch.device, weights_path: Path) -> None:
    model = ConditionalAE().to(device)
    opt = Adam(model.parameters(), lr=lr)

    print("\n[AE] Training")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            cond    = batch["cond"].to(device)
            year_t  = batch["year_target"].to(device)
            day_t   = batch["day_target"].to(device)
            year_oh = F.one_hot(year_t, 10).float()
            day_oh  = F.one_hot(day_t, 31).float()
            opt.zero_grad()
            yl, dl = model(cond, year_oh, day_oh)
            loss = ce_loss(yl, dl, year_t, day_t)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        val = _eval_ae(model, test_loader, device)
        print(f"  {epoch:03d}/{epochs}  train={train_loss/len(train_loader):.4f}  val={val:.4f}")

    torch.save(model.state_dict(), weights_path)
    print(f"[AE] saved → {weights_path}")


def _eval_ae(model: ConditionalAE, loader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            cond    = batch["cond"].to(device)
            year_t  = batch["year_target"].to(device)
            day_t   = batch["day_target"].to(device)
            year_oh = F.one_hot(year_t, 10).float()
            day_oh  = F.one_hot(day_t, 31).float()
            yl, dl = model(cond, year_oh, day_oh)
            total += ce_loss(yl, dl, year_t, day_t).item()
    return total / len(loader)


def train_cgan(train_loader, test_loader, epochs: int, lr: float,
               device: torch.device, weights_path: Path) -> None:
    G = CGANGenerator().to(device)
    D = CGANDiscriminator().to(device)
    opt_g = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = Adam(D.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    tau = 1.0

    print("\n[CGAN] Training")
    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        g_run = d_run = 0.0

        for batch in train_loader:
            cond   = batch["cond"].to(device)
            year_t = batch["year_target"].to(device)
            day_t  = batch["day_target"].to(device)
            B = cond.size(0)
            real_yr = F.one_hot(year_t, 10).float()
            real_dy = F.one_hot(day_t, 31).float()

            # discriminator step
            opt_d.zero_grad()
            noise = torch.randn(B, NOISE_DIM, device=device)
            with torch.no_grad():
                fy, fd = G(noise, cond)
            fy_s = F.gumbel_softmax(fy, tau=tau, hard=False)
            fd_s = F.gumbel_softmax(fd, tau=tau, hard=False)
            d_loss = 0.5 * (
                bce(D(cond, real_yr, real_dy), torch.full((B, 1), 0.9, device=device))
                + bce(D(cond, fy_s, fd_s), torch.full((B, 1), 0.1, device=device))
            )
            d_loss.backward()
            opt_d.step()

            # generator step
            opt_g.zero_grad()
            noise = torch.randn(B, NOISE_DIM, device=device)
            fy, fd = G(noise, cond)
            fy_s = F.gumbel_softmax(fy, tau=tau, hard=False)
            fd_s = F.gumbel_softmax(fd, tau=tau, hard=False)
            g_loss = (
                bce(D(cond, fy_s, fd_s), torch.ones(B, 1, device=device))
                + 0.5 * ce_loss(fy, fd, year_t, day_t)
            )
            g_loss.backward()
            opt_g.step()

            g_run += g_loss.item()
            d_run += d_loss.item()

        tau = max(0.3, tau * 0.98)
        print(f"  {epoch:03d}/{epochs}  G={g_run/len(train_loader):.4f}"
              f"  D={d_run/len(train_loader):.4f}  tau={tau:.3f}")

    torch.save({"generator": G.state_dict(), "discriminator": D.state_dict()}, weights_path)
    print(f"[CGAN] saved → {weights_path}")


def train_lstm(train_loader, test_loader, epochs: int, lr: float,
               device: torch.device, weights_path: Path) -> None:
    model = ConditionalLSTM().to(device)
    opt = Adam(model.parameters(), lr=lr)

    print("\n[LSTM] Training")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            day_tok    = batch["day_tok"].to(device)
            mon_tok    = batch["mon_tok"].to(device)
            leap_tok   = batch["leap_tok"].to(device)
            decade_tok = batch["decade_tok"].to(device)
            year_t     = batch["year_target"].to(device)
            day_t      = batch["day_target"].to(device)
            opt.zero_grad()
            yl, dl = model(day_tok, mon_tok, leap_tok, decade_tok)
            loss = ce_loss(yl, dl, year_t, day_t)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        val = _eval_tok(model, test_loader, device)
        print(f"  {epoch:03d}/{epochs}  train={train_loss/len(train_loader):.4f}  val={val:.4f}")

    torch.save(model.state_dict(), weights_path)
    print(f"[LSTM] saved → {weights_path}")


def train_transformer(train_loader, test_loader, epochs: int, lr: float,
                      device: torch.device, weights_path: Path) -> None:
    model = DateTransformer().to(device)
    opt = Adam(model.parameters(), lr=lr)

    print("\n[Transformer] Training")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            day_tok    = batch["day_tok"].to(device)
            mon_tok    = batch["mon_tok"].to(device)
            leap_tok   = batch["leap_tok"].to(device)
            decade_tok = batch["decade_tok"].to(device)
            year_t     = batch["year_target"].to(device)
            day_t      = batch["day_target"].to(device)
            opt.zero_grad()
            yl, dl = model(day_tok, mon_tok, leap_tok, decade_tok)
            loss = ce_loss(yl, dl, year_t, day_t)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        val = _eval_tok(model, test_loader, device)
        print(f"  {epoch:03d}/{epochs}  train={train_loss/len(train_loader):.4f}  val={val:.4f}")

    torch.save(model.state_dict(), weights_path)
    print(f"[Transformer] saved → {weights_path}")


def _eval_tok(model, loader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            day_tok    = batch["day_tok"].to(device)
            mon_tok    = batch["mon_tok"].to(device)
            leap_tok   = batch["leap_tok"].to(device)
            decade_tok = batch["decade_tok"].to(device)
            year_t     = batch["year_target"].to(device)
            day_t      = batch["day_target"].to(device)
            yl, dl = model(day_tok, mon_tok, leap_tok, decade_tok)
            total += ce_loss(yl, dl, year_t, day_t).item()
    return total / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ae", "cgan", "lstm", "transformer", "all"], default="all")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--data-path",  type=Path,  default=Path(__file__).parent.parent / "data" / "data.txt")
    parser.add_argument("--weights-dir",type=Path,  default=Path(__file__).parent / "weights")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    args.weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  data={args.data_path}  seed={args.seed}")

    train_loader, test_loader = get_dataloaders(args.data_path, batch_size=args.batch_size, seed=args.seed)
    print(f"train batches={len(train_loader)}  val batches={len(test_loader)}")

    W = args.weights_dir
    if args.model in ("ae",          "all"): train_ae(train_loader, test_loader, args.epochs, args.lr, device, W / "ae.pt")
    if args.model in ("cgan",        "all"): train_cgan(train_loader, test_loader, args.epochs, args.lr, device, W / "cgan.pt")
    if args.model in ("lstm",        "all"): train_lstm(train_loader, test_loader, args.epochs, args.lr, device, W / "lstm.pt")
    if args.model in ("transformer", "all"): train_transformer(train_loader, test_loader, args.epochs, args.lr, device, W / "transformer.pt")


if __name__ == "__main__":
    main()
