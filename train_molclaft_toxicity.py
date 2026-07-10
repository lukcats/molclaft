import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from molclaft import MolCLaFTConfig, MolCLaFTForToxicityPrediction
from molclaft.data import MolCLaFTToxicityDataset, collate_molclaft


def parse_args():
    parser = argparse.ArgumentParser(description="Train MolCLaFT for toxicity prediction.")
    parser.add_argument("--train_file", required=True, help="TSV/CSV file with SMILES and toxicity label columns.")
    parser.add_argument("--validation_file", default=None, help="Optional TSV/CSV validation file.")
    parser.add_argument("--smiles_column", default="SMILES")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--clm_name_or_path", default="laituan245/molt5-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--align_weight", type=float, default=0.1)
    parser.add_argument("--distill_weight", type=float, default=0.1)
    parser.add_argument("--output_path", default="molclaft_toxicity.pt")
    parser.add_argument("--unfreeze_clm", action="store_true")
    return parser.parse_args()


def make_loader(args, tokenizer, path, shuffle):
    dataset = MolCLaFTToxicityDataset(
        data_file=path,
        tokenizer=tokenizer,
        smiles_column=args.smiles_column,
        label_column=args.label_column,
        max_length=args.max_length,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_molclaft, pad_token_id=tokenizer.pad_token_id),
    )


def move_batch(batch, device):
    return batch.to(device)


def run_epoch(model, dataloader, optimizer, scheduler, device, train=True):
    model.train(train)
    totals = {"loss": 0.0, "pred_loss": 0.0, "align_loss": 0.0, "distill_loss": 0.0}

    for batch in dataloader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                atom_features=batch.atom_features,
                edge_index=batch.edge_index,
                graph_batch=batch.graph_batch,
                labels=batch.labels,
            )
            if train:
                optimizer.zero_grad()
                outputs["loss"].backward()
                optimizer.step()
                scheduler.step()

        for key in totals:
            totals[key] += float(outputs[key].detach().cpu())

    return {key: value / max(len(dataloader), 1) for key, value in totals.items()}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.clm_name_or_path, model_max_length=args.max_length)

    train_loader = make_loader(args, tokenizer, args.train_file, shuffle=True)
    val_loader = make_loader(args, tokenizer, args.validation_file, shuffle=False) if args.validation_file else None

    config = MolCLaFTConfig(
        clm_name_or_path=args.clm_name_or_path,
        num_labels=args.num_labels,
        align_weight=args.align_weight,
        distill_weight=args.distill_weight,
        freeze_clm=not args.unfreeze_clm,
    )
    model = MolCLaFTForToxicityPrediction(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    best_val = None
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, scheduler, device, train=True)
        print(f"epoch={epoch} train={train_metrics}")

        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, optimizer, scheduler, device, train=False)
            print(f"epoch={epoch} validation={val_metrics}")
            if best_val is None or val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save({"model": model.state_dict(), "config": config.__dict__}, args.output_path)

    if val_loader is None:
        torch.save({"model": model.state_dict(), "config": config.__dict__}, args.output_path)


if __name__ == "__main__":
    main()
