# train.py
import os
import random
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

import config
from model_arch import (
    load_embeddings,
    build_tokenizer_and_model,
    apply_peft,
    VisionProjector,
    save_checkpoint,
)
from dataset import load_examples, VisionPrefixDataset

try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TENSORBOARD = True
except ImportError:
    USE_TENSORBOARD = False


def setup_logging():
    """Create log directory, CSV logger, and optional TensorBoard writer."""
    log_dir = os.path.join(config.OUTPUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    csv_path = os.path.join(log_dir, f"train_log_{timestamp}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "step", "loss", "avg_loss"])

    tb_dir = os.path.join(log_dir, "tensorboard", timestamp)
    tb_writer = SummaryWriter(log_dir=tb_dir) if USE_TENSORBOARD else None

    print(f"Logging to: {csv_path}")
    if tb_writer:
        print(f"TensorBoard logs: {tb_dir}")

    return csv_file, csv_writer, tb_writer


def main():
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    csv_file, csv_writer, tb_writer = setup_logging()

    print("Loading embeddings...")
    image_ids, embeddings = load_embeddings(config.EMB_FILE)
    print("Emb shape:", embeddings.shape)
    imageid_to_idx = {os.path.basename(name): i for i, name in enumerate(image_ids)}

    examples = load_examples(config.DATA_JSONL, imageid_to_idx)
    print("Usable examples:", len(examples))
    if len(examples) == 0:
        raise RuntimeError("No examples matched embeddings; check filenames.")

    tokenizer, model = build_tokenizer_and_model(config.MODEL_ID, config)
    print("Applying PEFT LoRA...")
    model = apply_peft(model, config)
    model.print_trainable_parameters()

    V_DIM = embeddings.shape[1]
    T_DIM = model.config.hidden_size
    print(f"V_DIM={V_DIM}, T_DIM={T_DIM}")
    projector = VisionProjector(
        V_DIM, T_DIM,
        hidden=min(2048, max(512, V_DIM * 2)),
        n_prefix_tokens=config.PREFIX_TOKENS
    ).to(config.DEVICE)

    dataset = VisionPrefixDataset(
        examples, embeddings, imageid_to_idx=imageid_to_idx, tokenizer=tokenizer,
        max_text_len=config.MAX_TEXT_LEN, max_label_len=config.MAX_LABEL_LEN
    )
    dataloader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS
    )
    # print("dataset and dataloader ready.")
    # print(f"Total training steps per epoch: {len(dataloader)}")
    # print(f"dataset: {dataset}")
    # print(f"dataloader batch: {next(iter(dataloader))}")
    # print("End of setup. Starting training...")
    peft_params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(peft_params + list(projector.parameters()), lr=config.LR)

    model.train()
    projector.train()
    print("Start training on device:", config.DEVICE)

    global_step = 0

    for epoch in range(config.EPOCHS):
        running_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            vision_emb = batch["vision_emb"].to(config.DEVICE, dtype=torch.float32)
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)

            B = input_ids.size(0)
            proj = projector(vision_emb)
            inputs_embeds = model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([proj, inputs_embeds], dim=1)

            prefix_mask = torch.ones((B, proj.size(1)), device=config.DEVICE, dtype=attention_mask.dtype)
            extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            prefix_labels = torch.full((B, proj.size(1)), -100, device=config.DEVICE, dtype=labels.dtype)
            extended_labels = torch.cat([prefix_labels, labels], dim=1)

            # outputs = model(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask, labels=extended_labels)
            
            # ensure seq lengths align
            min_len = min(inputs_embeds.size(1), extended_attention_mask.size(1), extended_labels.size(1))
            inputs_embeds = inputs_embeds[:, :min_len, :]
            extended_attention_mask = extended_attention_mask[:, :min_len]
            extended_labels = extended_labels[:, :min_len]
            
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_attention_mask,
                labels=extended_labels
            )
            loss = outputs.loss

            if torch.isnan(loss):
                print(f"[WARN] NaN loss detected at step {global_step}. Skipping update.")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            # Log every PRINT_EVERY steps
            if step % config.PRINT_EVERY == 0:
                avg_loss = running_loss / config.PRINT_EVERY
                print(f"[Epoch {epoch+1}] Step {step}/{len(dataloader)} — avg loss: {avg_loss:.4f}")
                csv_writer.writerow([epoch + 1, global_step, loss.item(), avg_loss])
                csv_file.flush()

                if tb_writer:
                    tb_writer.add_scalar("train/loss_step", loss.item(), global_step)
                    tb_writer.add_scalar("train/avg_loss", avg_loss, global_step)
                    tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

                running_loss = 0.0

        # Save checkpoint per epoch
        ckpt_dir = os.path.join(config.OUTPUT_DIR, f"epoch_{epoch+1}")
        save_checkpoint(projector, model, ckpt_dir)
        print(f"Saved checkpoint to {ckpt_dir}")

        if tb_writer:
            tb_writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)

    csv_file.close()
    if tb_writer:
        tb_writer.close()

    print("Training complete.")


if __name__ == "__main__":
    main()
