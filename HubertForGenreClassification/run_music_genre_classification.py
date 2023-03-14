
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import tqdm
from transformers import AutoFeatureExtractor

from MusicGenreClassificationModel import HubertForGenreClassification
from utils import get_device, load_split_dataframe, create_dataset_w_dataframe


def setup_dataloader(args, feature_extractor):
    # Create dataframe based on input data split
    train_dataframe, dev_dataframe, test_dataframe = load_split_dataframe(args.data_split_txt_filepath)

    # Create dataset
    train_dataset = create_dataset_w_dataframe(train_dataframe, args.data_dir, feature_extractor, args.sample_rate)
    dev_dataset = create_dataset_w_dataframe(dev_dataframe, args.data_dir, feature_extractor, args.sample_rate)
    test_dataset = create_dataset_w_dataframe(test_dataframe, args.data_dir, feature_extractor, args.sample_rate)

    # Create dataloader
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    return train_loader, dev_loader, test_loader


def setup_optimizer(args, model, device):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters())
    return criterion, optimizer

def setup_model(args):
    model = HubertForGenreClassification(args.model_name_or_path)
    return model


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    if training:
        model.train()
    else:
        model.eval()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for loaded_inputs in tqdm.tqdm(loader):
        # put model inputs to device
        inputs = loaded_inputs["input_values"][0]
        labels = loaded_inputs["label"]

        inputs, labels = inputs.to(device).float(), labels.to(device).long()

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs)

        # calculate prediction loss
        loss = criterion(pred_logits.squeeze(), labels)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        preds = pred_logits.argmax(-1)
        pred_labels.extend(preds.cpu().numpy())
        target_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(pred_labels, target_labels)
    epoch_loss /= len(loader)

    return epoch_loss, acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def main(args):
    device = get_device(args.force_cpu)

    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)

    # get data loaders
    train_loader, dev_loader, test_loader = setup_dataloader(args, feature_extractor)
    loaders = {"train": train_loader, "val": dev_loader, "test": test_loader}

    # build model
    model = setup_model(args).to(device)
    print(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model, device)

    all_train_acc = []
    all_train_loss = []
    all_val_acc = []
    all_val_loss = []

    best_val_acc = 0
    best_val_epoch = -1

    if args.do_train:
        for epoch in range(args.num_epochs):
            # train model for a single epoch
            print(f"Epoch {epoch}")
            train_loss, train_acc = train_epoch(
                args,
                model,
                loaders["train"],
                optimizer,
                criterion,
                device,
            )

            print(f"train loss : {train_loss} | train acc: {train_acc}")
            all_train_acc.append(train_acc)
            all_train_loss.append(train_loss)

            if epoch % args.val_every == 0:
                val_loss, val_acc = validate(
                    args,
                    model,
                    loaders["val"],
                    optimizer,
                    criterion,
                    device,
                )
                print(f"val loss : {val_loss} | val acc: {val_acc}")
                all_val_acc.append(val_acc)
                all_val_loss.append(val_loss)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_epoch = epoch
                    Path(args.outputs_dir).mkdir(parents=True, exist_ok=True)
                    ckpt_model_file = os.path.join(args.outputs_dir, "model.ckpt")
                    performance_file = os.path.join(args.outputs_dir, "results.txt")
                    print("saving model to ", ckpt_model_file)
                    torch.save(model, ckpt_model_file)
                    open(performance_file, 'w').write(f"Best epoch: {best_val_epoch} | Accuracy: {best_val_acc}")
    elif args.do_eval:
        # Load pretrained model
        model = torch.load(os.path.join(args.outputs_dir, "model.ckpt"))
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            criterion,
            device,
        )
        print(f"val loss : {val_loss} | val acc: {val_acc}")

    if args.do_test:
        if not args.do_train:
            # Load pretrained model
            model = torch.load(os.path.join(args.outputs_dir, "model.ckpt"))
        test_loss, test_acc = validate(
            args,
            model,
            loaders["test"],
            optimizer,
            criterion,
            device,
        )
        print(f"test loss : {test_loss} | test acc: {test_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true", help="training mode")
    parser.add_argument("--do_eval", action="store_true", help="training mode")
    parser.add_argument("--do_test", action="store_true", help="training mode")

    parser.add_argument("--outputs_dir", type=str, default="output", help="where to save training outputs")
    parser.add_argument("--data_dir", type=str, help="where the music audio files are stored")
    parser.add_argument("--data_split_txt_filepath", type=str, default="../data_split.txt",
                        help="where the data split txt file is stored")

    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--num_epochs", default=30, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=2,
        type=int,
        help="number of epochs between every eval loop",
    )
    # parser.add_argument(
    #     "--save_every",
    #     default=5,
    #     type=int,
    #     help="number of epochs between saving model checkpoint",
    # )

    parser.add_argument(
        "--model_name_or_path",
        default="facebook/hubert-base-ls960",
        type=str,
        help=""
    )

    parser.add_argument(
        "--sample_rate",
        default=16000,
        type=int,
        help="max length for processing audio",
    )

    args = parser.parse_args()
    main(args)
