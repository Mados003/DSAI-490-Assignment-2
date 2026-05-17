import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from utils.dataset import get_dataloaders
from utils.date_utils import DAYS, MONTHS, check_conditions, format_date, snap_to_valid
from utils.metrics import condition_satisfaction_rate
from utils.tokenizer import MIN_DECADE
from models.ae import ConditionalAE
from models.cgan import CGANGenerator, NOISE_DIM
from models.lstm import ConditionalLSTM
from models.transformer import DateTransformer


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _decode_batch(yl: torch.Tensor, dl: torch.Tensor, batch: dict) -> list[tuple[int, int, int]]:
    year_preds = yl.argmax(dim=1).cpu().tolist()
    day_preds  = dl.argmax(dim=1).cpu().tolist()
    results = []
    for i in range(len(year_preds)):
        decade    = batch["decade_tok"][i].item() + MIN_DECADE
        month_str = MONTHS[batch["mon_tok"][i].item()]
        req_day   = DAYS[batch["day_tok"][i].item()]
        results.append(snap_to_valid(decade * 10 + year_preds[i], day_preds[i] + 1, month_str, decade, req_day))
    return results


def predict_ae(model: ConditionalAE, batch: dict, device: torch.device) -> list:
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        yl, dl = model.generate(batch["cond"].to(device))
    return _decode_batch(yl, dl, batch)


def predict_cgan(G: CGANGenerator, batch: dict, device: torch.device) -> list:
    G.eval()
    cond  = batch["cond"].to(device)
    noise = torch.randn(cond.size(0), NOISE_DIM, device=device)
    with torch.no_grad():
        yl, dl = G(noise, cond)
    return _decode_batch(yl, dl, batch)


def predict_lstm(model: ConditionalLSTM, batch: dict, device: torch.device) -> list:
    model.eval()
    with torch.no_grad():
        yl, dl = model(
            batch["day_tok"].to(device), batch["mon_tok"].to(device),
            batch["leap_tok"].to(device), batch["decade_tok"].to(device),
        )
    return _decode_batch(yl, dl, batch)


def predict_transformer(model: DateTransformer, batch: dict, device: torch.device) -> list:
    model.eval()
    with torch.no_grad():
        yl, dl = model(
            batch["day_tok"].to(device), batch["mon_tok"].to(device),
            batch["leap_tok"].to(device), batch["decade_tok"].to(device),
        )
    return _decode_batch(yl, dl, batch)


def collect(predict_fn, test_loader, device: torch.device) -> tuple[list, list]:
    preds: list[tuple[int, int, int]] = []
    conds: list[tuple[str, str, bool, int]] = []
    for batch in test_loader:
        preds.extend(predict_fn(batch, device))
        for i in range(batch["day_tok"].size(0)):
            conds.append((
                DAYS[batch["day_tok"][i].item()],
                MONTHS[batch["mon_tok"][i].item()],
                bool(batch["leap_tok"][i].item()),
                batch["decade_tok"][i].item() + MIN_DECADE,
            ))
    return preds, conds


def show_examples(name: str, preds: list, conds: list, n: int = 5) -> None:
    ok   = [(p, c) for p, c in zip(preds, conds) if all(check_conditions(*p, *c).values())]
    fail = [(p, c) for p, c in zip(preds, conds) if not all(check_conditions(*p, *c).values())]
    print(f"\n{name}")
    print(f"  successes (up to {n}):")
    for (d, m, y), (rd, rm, rl, rdec) in ok[:n]:
        print(f"    [{rd}][{rm}][{rl}][{rdec}] -> {format_date(d, m, y)}  {check_conditions(d, m, y, rd, rm, rl, rdec)}")
    print(f"  failures (up to {n}):")
    for (d, m, y), (rd, rm, rl, rdec) in fail[:n]:
        print(f"    [{rd}][{rm}][{rl}][{rdec}] -> {format_date(d, m, y)}  {check_conditions(d, m, y, rd, rm, rl, rdec)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ae", "cgan", "lstm", "transformer", "all"], default="all")
    parser.add_argument("--data-path",   type=Path, default=Path(__file__).parent.parent / "data" / "data.txt")
    parser.add_argument("--weights-dir", type=Path, default=Path(__file__).parent / "weights")
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--batch-size",  type=int,  default=256)
    args = parser.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    _, test_loader = get_dataloaders(args.data_path, batch_size=args.batch_size, seed=args.seed)

    W   = args.weights_dir
    run = args.model

    header = f"{'model':<15} {'all':>7} {'day':>7} {'month':>7} {'leap':>7} {'decade':>7}"
    print(header)
    print("-" * len(header))

    def report(name: str, csr: dict) -> None:
        print(f"  {name:<13} {csr['all']:>7.1%} {csr['day_of_week']:>7.1%}"
              f" {csr['month']:>7.1%} {csr['leap_year']:>7.1%} {csr['decade']:>7.1%}")

    if run in ("ae", "all"):
        wp = W / "ae.pt"
        if not wp.exists():
            print("  [AE] not trained")
        else:
            m = ConditionalAE()
            m.load_state_dict(torch.load(wp, map_location=device)); m.to(device)
            preds, conds = collect(lambda b, d: predict_ae(m, b, d), test_loader, device)
            report("AE", condition_satisfaction_rate(preds, conds))
            show_examples("AE", preds, conds)

    if run in ("cgan", "all"):
        wp = W / "cgan.pt"
        if not wp.exists():
            print("  [CGAN] not trained")
        else:
            G = CGANGenerator()
            G.load_state_dict(torch.load(wp, map_location=device)["generator"]); G.to(device)
            preds, conds = collect(lambda b, d: predict_cgan(G, b, d), test_loader, device)
            report("CGAN", condition_satisfaction_rate(preds, conds))
            show_examples("CGAN", preds, conds)

    if run in ("lstm", "all"):
        wp = W / "lstm.pt"
        if not wp.exists():
            print("  [LSTM] not trained")
        else:
            m = ConditionalLSTM()
            m.load_state_dict(torch.load(wp, map_location=device)); m.to(device)
            preds, conds = collect(lambda b, d: predict_lstm(m, b, d), test_loader, device)
            report("LSTM", condition_satisfaction_rate(preds, conds))
            show_examples("LSTM", preds, conds)

    if run in ("transformer", "all"):
        wp = W / "transformer.pt"
        if not wp.exists():
            print("  [Transformer] not trained")
        else:
            m = DateTransformer()
            m.load_state_dict(torch.load(wp, map_location=device)); m.to(device)
            preds, conds = collect(lambda b, d: predict_transformer(m, b, d), test_loader, device)
            report("Transformer", condition_satisfaction_rate(preds, conds))
            show_examples("Transformer", preds, conds)


if __name__ == "__main__":
    main()
