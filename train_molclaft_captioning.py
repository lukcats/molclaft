import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from molclaft.captioning import InContextCaptionDataset, collate_captioning, compute_icmc_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MolT5 with MolCLaFT in-context molecule captioning.")
    parser.add_argument("--train_file", default="ChEBI-20_data/train.txt")
    parser.add_argument("--validation_file", default="ChEBI-20_data/validation.txt")
    parser.add_argument("--smiles_column", default="SMILES")
    parser.add_argument("--caption_column", default="description")
    parser.add_argument("--model_name_or_path", default="laituan245/molt5-small")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--lambda_context", type=float, default=0.2)
    parser.add_argument("--output_path", default="molclaft_icmc.pt")
    return parser.parse_args()


def make_loader(args, tokenizer, path, shuffle):
    dataset = InContextCaptionDataset(
        data_file=path,
        tokenizer=tokenizer,
        smiles_column=args.smiles_column,
        caption_column=args.caption_column,
        top_k=args.top_k,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_captioning, pad_token_id=tokenizer.pad_token_id),
    )


def run_epoch(model, dataloader, optimizer, scheduler, device, lambda_context, train=True):
    model.train(train)
    total_loss = 0.0
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(train):
            loss = compute_icmc_loss(model, batch, lambda_context=lambda_context)["loss"]
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(dataloader), 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)

    train_loader = make_loader(args, tokenizer, args.train_file, shuffle=True)
    val_loader = make_loader(args, tokenizer, args.validation_file, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * len(train_loader),
    )

    best_val = None
    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model, train_loader, optimizer, scheduler, device, args.lambda_context, train=True
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, scheduler, device, args.lambda_context, train=False
        )
        print(f"epoch={epoch} train_loss={train_loss:.4f} validation_loss={val_loss:.4f}")
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.output_path)


if __name__ == "__main__":
    main()
