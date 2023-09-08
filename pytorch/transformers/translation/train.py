from pathlib import Path  # returns absolute path, given relative path

import torch
import torch.nn as nn
from config import get_config, get_weights_file_path
from dataset import BilingualDataset, causal_mask
from datasets import load_dataset
from model import build_transformer

# bpe tokenizer, word level tokenizer, sub-word level tokenizer, word-part tokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate
from tqdm import tqdm


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_src.token_to_id("[EOS]")

    # precompute the ecoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # initialize the decodre input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:  # first dimension is batch
            break

        # build the mask for the target (decoder input)
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next token, that is the next token of the last token given to the encoder
        prob = model.project(out[:, -1])

        # select the token with the max probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    writer,
    num_examples=2,
):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # size of the control window (just use default value)
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # during inference, we want to calculate the encoder_output only once and reuse for every every token
            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg("-" * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()


def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(config):
    dataset_raw = load_dataset(
        "opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train"
    )

    # build the tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config["lang_tgt"])

    train_dataset_size = int(0.9 * len(dataset_raw))
    val_dataset_size = len(dataset_raw) - train_dataset_size

    train_dataset_raw, val_dataset_raw = random_split(
        dataset_raw, [train_dataset_size, val_dataset_size]
    )

    train_dataset = BilingualDataset(
        train_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_dataset = BilingualDataset(
        val_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=True
    )  # we want to process each sentence one-by-one

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )


def train_model(config):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {DEVICE}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(DEVICE)

    # tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(
        DEVICE
    )  # we want to ignore the padding to contribute to the loss

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(DEVICE)  # (b, seq_len)
            decoder_input = batch["decoder_input"].to(DEVICE)  # (b, seq_len)

            # why these two masks are different is because, in one case we are hiding the padding tokens and in the other we also hide the words that come after the sequence
            encoder_mask = batch["encoder_mask"].to(DEVICE)  # (b, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(DEVICE)  # (b, 1, seq_len, seq_len)

            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (b, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (b, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (b, seq_len, tgt_vocab_size)

            label = batch["label"].to(DEVICE)  # (b, seq_len)

            # (b, seq_len, tgt_vocab_size) -> (b * seq_len, tgt_vocab_size)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad(
                set_to_none=True
            )  # reset the gradients to None instead of filling them with a tensor of zeroes

            global_step += 1

        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            DEVICE,
            lambda msg: batch_iterator.write(msg),
            global_step,
            writer,
        )

    # save the model at the end of every epoch
    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        },
        model_filename,
    )  # save the optimizer state, it is quite big but resumeable is a good feature to have and not let the optimizer start from 0


if __name__ == "__main__":
    config = get_config()
    train_model(config)
