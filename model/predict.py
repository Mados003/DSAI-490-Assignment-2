import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from utils.date_utils import snap_to_valid
from utils.tokenizer import MIN_DECADE, DAYS_IDX, MONTHS_IDX, encode_conditions, format_output_line, parse_condition_line
from models.ae import ConditionalAE
from models.cgan import CGANGenerator, NOISE_DIM
from models.lstm import ConditionalLSTM
from models.transformer import DateTransformer


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _snap(yl: torch.Tensor, dl: torch.Tensor,
          day_str: str, month_str: str, decade: int) -> tuple[int, int, int]:
    year_full    = decade * 10 + yl.argmax(dim=1).item()
    day_1indexed = dl.argmax(dim=1).item() + 1
    return snap_to_valid(year_full, day_1indexed, month_str, decade, day_str)


def run_ae(model: ConditionalAE, day_str: str, month_str: str,
           leap: bool, decade: int, device: torch.device) -> tuple[int, int, int]:
    cond = encode_conditions(day_str, month_str, leap, decade).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        yl, dl = model.generate(cond)
    return _snap(yl, dl, day_str, month_str, decade)


def run_cgan(G: CGANGenerator, day_str: str, month_str: str,
             leap: bool, decade: int, device: torch.device) -> tuple[int, int, int]:
    cond  = encode_conditions(day_str, month_str, leap, decade).unsqueeze(0).to(device)
    noise = torch.randn(1, NOISE_DIM, device=device)
    G.eval()
    with torch.no_grad():
        yl, dl = G(noise, cond)
    return _snap(yl, dl, day_str, month_str, decade)


def run_lstm(model: ConditionalLSTM, day_str: str, month_str: str,
             leap: bool, decade: int, device: torch.device) -> tuple[int, int, int]:
    day_tok    = torch.tensor([DAYS_IDX[day_str]],        dtype=torch.long, device=device)
    mon_tok    = torch.tensor([MONTHS_IDX[month_str]],    dtype=torch.long, device=device)
    leap_tok   = torch.tensor([1 if leap else 0],         dtype=torch.long, device=device)
    decade_tok = torch.tensor([decade - MIN_DECADE],      dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        yl, dl = model(day_tok, mon_tok, leap_tok, decade_tok)
    return _snap(yl, dl, day_str, month_str, decade)


def run_transformer(model: DateTransformer, day_str: str, month_str: str,
                    leap: bool, decade: int, device: torch.device) -> tuple[int, int, int]:
    day_tok    = torch.tensor([DAYS_IDX[day_str]],        dtype=torch.long, device=device)
    mon_tok    = torch.tensor([MONTHS_IDX[month_str]],    dtype=torch.long, device=device)
    leap_tok   = torch.tensor([1 if leap else 0],         dtype=torch.long, device=device)
    decade_tok = torch.tensor([decade - MIN_DECADE],      dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        yl, dl = model(day_tok, mon_tok, leap_tok, decade_tok)
    return _snap(yl, dl, day_str, month_str, decade)


def load_model(model_name: str, weights_dir: Path, device: torch.device):
    wp = weights_dir / f"{model_name}.pt"
    if not wp.exists():
        print(f"weights not found at {wp} — run: python train.py --model {model_name}")
        sys.exit(1)
    if model_name == "ae":
        m = ConditionalAE()
        m.load_state_dict(torch.load(wp, map_location=device))
        return m.to(device)
    if model_name == "cgan":
        G = CGANGenerator()
        G.load_state_dict(torch.load(wp, map_location=device)["generator"])
        return G.to(device)
    if model_name == "lstm":
        m = ConditionalLSTM()
        m.load_state_dict(torch.load(wp, map_location=device))
        return m.to(device)
    if model_name == "transformer":
        m = DateTransformer()
        m.load_state_dict(torch.load(wp, map_location=device))
        return m.to(device)
    raise ValueError(model_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",   required=True, type=Path)
    parser.add_argument("-o", "--output",  required=True, type=Path)
    parser.add_argument("--model",         default="ae", choices=["ae", "cgan", "lstm", "transformer"])
    parser.add_argument("--weights-dir",   type=Path, default=Path(__file__).parent / "weights")
    args = parser.parse_args()

    device = get_device()
    model  = load_model(args.model, args.weights_dir, device)

    run = {
        "ae":          lambda d, m, l, dec: run_ae(model, d, m, l, dec, device),
        "cgan":        lambda d, m, l, dec: run_cgan(model, d, m, l, dec, device),
        "lstm":        lambda d, m, l, dec: run_lstm(model, d, m, l, dec, device),
        "transformer": lambda d, m, l, dec: run_transformer(model, d, m, l, dec, device),
    }[args.model]

    lines = [ln.strip() for ln in args.input.read_text().splitlines() if ln.strip()]
    out   = []
    for line in lines:
        day_str, month_str, leap, decade = parse_condition_line(line)
        day, month_num, year = run(day_str, month_str, leap, decade)
        out.append(format_output_line(day_str, month_str, leap, decade, day, month_num, year))

    args.output.write_text("\n".join(out) + "\n")
    print(f"wrote {len(out)} predictions → {args.output}")


if __name__ == "__main__":
    main()
