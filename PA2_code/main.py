import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from utilities import Utilities

# transformer.py should define these
from transformer import Encoder, EncoderClassifier, DecoderLM


# ----------------------------
# Reproducibility / Device
# ----------------------------
seed = 42
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Helpers: data + batching
# ----------------------------
def load_texts(directory):
    """Load all texts from directory ignoring files with 'test' in the name."""
    texts = []
    for filename in os.listdir(directory):
        if "test" in filename:
            continue
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def collate_batch_fixed_block(batch, block_size):
    """
    Collate (input_ids, label) from SpeechesClassificationDataset into:
      X: (B, block_size) padded/truncated
      Y: (B,)
    """
    data, labels = zip(*batch)
    padded = pad_sequence(data, batch_first=True, padding_value=0)  # (B, T_var)
    padded = padded[:, :block_size]
    padded = torch.nn.functional.pad(
        padded, (0, max(0, block_size - padded.shape[1])), "constant", 0
    )
    labels = torch.stack(labels)
    return padded, labels


def make_lm_loader(tokenizer, path, block_size, batch_size, shuffle=False):
    """Build a LanguageModelingDataset loader from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    ds = LanguageModelingDataset(tokenizer, txt, block_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def infinite_loader(dataloader):
    """Yield batches forever (so we can run exactly max_iters)."""
    while True:
        for batch in dataloader:
            yield batch


# ----------------------------
# Metrics
# ----------------------------
def compute_classifier_accuracy(classifier, data_loader):
    classifier.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            logits = classifier(X)  # (B,3)
            pred = torch.argmax(logits, dim=1)
            total_correct += (pred == Y).sum().item()
            total_samples += Y.size(0)
    classifier.train()
    return 100.0 * total_correct / max(1, total_samples)


def compute_perplexity(decoder, data_loader, eval_iters=200):
    """
    Perplexity = exp(mean cross entropy).
    Assumes decoder(X, Y) returns CE loss scalar.
    """
    decoder.eval()
    losses = []
    with torch.no_grad():
        for i, (X, Y) in enumerate(data_loader):
            if i >= eval_iters:
                break
            X, Y = X.to(device), Y.to(device)
            loss = decoder(X, Y)
            losses.append(loss.item())
    decoder.train()
    mean_loss = torch.tensor(losses).mean()
    return torch.exp(mean_loss).item()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------
# Part 1: Encoder + Classifier
# ----------------------------
def run_part1(args, tokenizer):
    print("\n========== [Part 1] Encoder + Classifier ==========")

    train_ds = SpeechesClassificationDataset(tokenizer, os.path.join(args.data_dir, "train_CLS.tsv"))
    test_ds  = SpeechesClassificationDataset(tokenizer, os.path.join(args.data_dir, "test_CLS.tsv"))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch_fixed_block(b, args.block_size)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch_fixed_block(b, args.block_size)
    )

    encoder = Encoder(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        n_hidden=args.ffn_hidden,
        pad_id=args.pad_id,
        dropout=args.dropout,
    ).to(device)

    classifier = EncoderClassifier(
        encoder=encoder,
        n_hidden=args.cls_hidden,
        n_output=args.n_output,
        pad_id=args.pad_id,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    print("[Part 1] Encoder parameters:", count_params(encoder))

    if args.sanity_check:
        print("\n[Part 1] Sanity check attention maps (encoder)...")
        encoder_cpu = encoder.to("cpu")
        encoder_cpu.eval()
        util = Utilities(tokenizer, encoder_cpu)
        util.sanity_check(args.sanity_sentence, args.block_size)
        encoder = encoder_cpu.to(device)
        classifier.encoder = encoder

    print("\n[Part 1] Training...")
    for epoch in range(args.epochs_cls):
        classifier.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = classifier(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_acc = compute_classifier_accuracy(classifier, train_loader)
        test_acc = compute_classifier_accuracy(classifier, test_loader)

        print(f"Epoch {epoch+1:02d}/{args.epochs_cls} | "
              f"loss={total_loss/len(train_loader):.4f} | "
              f"train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}%")

    print("\n[Part 1] Encoder parameters:", count_params(classifier.encoder))
    print("[Part 1] Final test accuracy:", f"{test_acc:.2f}%")

    return classifier, encoder


# ----------------------------
# Part 2: Decoder Language Model
# ----------------------------
def run_part2(args, tokenizer):
    # wrapper: use args.pos_encoding
    return run_part2_with_posenc(args, tokenizer, pos_encoding=args.pos_encoding)


def run_part2_with_posenc(args, tokenizer, pos_encoding: str):
    """
    Run Part 2 training/eval using a specific positional encoding setting.
    This is used by:
      - --run part2  (single setting)
      - --run part3  (sweep learned/none/alibi)
    """
    print("\n========== [Part 2] Decoder Language Model ==========")
    print(f"[Part 2] pos_encoding = {pos_encoding}")

    train_lm_path = os.path.join(args.data_dir, "train_LM.txt")
    train_lm_loader = make_lm_loader(
        tokenizer, train_lm_path, args.block_size, args.batch_size, shuffle=True
    )

    test_obama = make_lm_loader(tokenizer, os.path.join(args.data_dir, "test_LM_obama.txt"),
                                args.block_size, args.batch_size, shuffle=False)
    test_wbush = make_lm_loader(tokenizer, os.path.join(args.data_dir, "test_LM_wbush.txt"),
                                args.block_size, args.batch_size, shuffle=False)
    test_ghbush = make_lm_loader(tokenizer, os.path.join(args.data_dir, "test_LM_hbush.txt"),
                                 args.block_size, args.batch_size, shuffle=False)

    decoder = DecoderLM(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        n_hidden=args.ffn_hidden,
        dropout=args.dropout,
        pad_id=args.pad_id,
        pos_encoding=pos_encoding,
    ).to(device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    print("[Part 2] Decoder parameters:", count_params(decoder))

    if args.sanity_check:
        print("\n[Part 2] Sanity check attention maps (decoder)...")
        class DecoderForUtils(nn.Module):
            def __init__(self, dec):
                super().__init__()
                self.dec = dec
            def forward(self, x):
                logits, attn_maps = self.dec(x, return_attn=True)
                return logits, attn_maps

        dec_cpu = DecoderForUtils(decoder.to("cpu"))
        dec_cpu.eval()
        util = Utilities(tokenizer, dec_cpu)
        util.sanity_check(args.sanity_sentence, args.block_size)
        decoder = dec_cpu.dec.to(device)

    print("\n[Part 2] Pretraining...")
    it = infinite_loader(train_lm_loader)
    decoder.train()

    history = []  # store (iter, train_ppl, obama, wbush, ghbush)

    for step in range(1, args.max_iters + 1):
        xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        loss = decoder(xb, yb)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_iters:
            train_ppl = compute_perplexity(decoder, train_lm_loader, eval_iters=args.eval_iters)
            obama_ppl = compute_perplexity(decoder, test_obama, eval_iters=args.eval_iters)
            wbush_ppl = compute_perplexity(decoder, test_wbush, eval_iters=args.eval_iters)
            ghbush_ppl = compute_perplexity(decoder, test_ghbush, eval_iters=args.eval_iters)

            history.append((step, train_ppl, obama_ppl, wbush_ppl, ghbush_ppl))

            print(f"Iter {step:03d}/{args.max_iters} | "
                  f"loss={loss.item():.4f} | "
                  f"train_ppl={train_ppl:.2f} | "
                  f"obama={obama_ppl:.2f} | wbush={wbush_ppl:.2f} | ghbush={ghbush_ppl:.2f}")

    final_train = compute_perplexity(decoder, train_lm_loader, eval_iters=args.eval_iters)
    final_obama = compute_perplexity(decoder, test_obama, eval_iters=args.eval_iters)
    final_wbush = compute_perplexity(decoder, test_wbush, eval_iters=args.eval_iters)
    final_ghbush = compute_perplexity(decoder, test_ghbush, eval_iters=args.eval_iters)

    print("\n[Part 2] Final PPL | train:", f"{final_train:.2f}",
          "| obama:", f"{final_obama:.2f}",
          "| wbush:", f"{final_wbush:.2f}",
          "| ghbush:", f"{final_ghbush:.2f}")

    return decoder, history, (final_train, final_obama, final_wbush, final_ghbush)


# ----------------------------
# Part 3: Positional Encoding Sweep (learned vs none vs alibi)
# ----------------------------
def run_part3(args, tokenizer):
    print("\n========== [Part 3] Architectural Exploration ==========")
    print("We will run Part 2 training three times and compare perplexity:")
    print("  1) learned (baseline)")
    print("  2) none    (no positional encoding)")
    print("  3) alibi   (AliBi positional bias)\n")

    results = {}

    for pe in ["learned", "none", "alibi"]:
        # run a fresh training for each setting
        decoder, history, finals = run_part2_with_posenc(args, tokenizer, pos_encoding=pe)
        results[pe] = {
            "params": count_params(decoder),
            "history": history,
            "finals": finals,
        }

    print("\n========== [Part 3] Summary Table ==========")
    print("pos_encoding | params | final_train | final_obama | final_wbush | final_ghbush")
    for pe in ["learned", "none", "alibi"]:
        params = results[pe]["params"]
        ft, fo, fw, fg = results[pe]["finals"]
        print(f"{pe:10s} | {params:6d} | {ft:10.2f} | {fo:10.2f} | {fw:10.2f} | {fg:11.2f}")

    #print("\nTip: in your report, you can also copy the intermediate PPL at "
    #      f"iters {args.eval_interval}, {2*args.eval_interval}, ... from the logs above.")
    return results


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--run", type=str, default="part1",
                   choices=["part1", "part2", "part3", "all"],
                   help="Which part to run.")

    p.add_argument("--data_dir", type=str, default="speechesdataset")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--ffn_hidden", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pad_id", type=int, default=0)

    p.add_argument("--epochs_cls", type=int, default=15)
    p.add_argument("--cls_hidden", type=int, default=100)
    p.add_argument("--n_output", type=int, default=3)

    p.add_argument("--max_iters", type=int, default=500)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_iters", type=int, default=200)

    p.add_argument("--sanity_check", action="store_true",
                   help="Run attention sanity check and save heatmaps.")
    p.add_argument("--sanity_sentence", type=str,
                   default="We will always meet the challenges of our time .")

    # Part 3 hook
    p.add_argument("--pos_encoding", type=str, default="learned",
                   choices=["learned", "none", "alibi"],
                   help="Positional encoding type.")

    return p.parse_args()


def main():
    args = parse_args()

    print("Loading data and creating tokenizer ...")
    texts = load_texts(args.data_dir)
    tokenizer = SimpleTokenizer(" ".join(texts))
    print("Vocabulary size is", tokenizer.vocab_size)
    print("Device:", device)

    if args.run == "part1":
        run_part1(args, tokenizer)
    elif args.run == "part2":
        run_part2(args, tokenizer)
    elif args.run == "part3":
        run_part3(args, tokenizer)
    else:  # all
        run_part1(args, tokenizer)
        run_part2(args, tokenizer)


if __name__ == "__main__":
    main()
