
import argparse

from torch.utils.data import DataLoader

from utils import get_device, load_split_dataframe, create_dataset_w_dataframe


def setup_dataloader(args, feature_extractor):
    # Create dataframe based on input data split
    train_dataframe, dev_dataframe, test_dataframe = load_split_dataframe(args.data_split_txt_filepath)

    # Create dataset
    train_dataset = create_dataset_w_dataframe(train_dataframe, args.data_dir, feature_extractor, args.max_length)
    dev_dataset = create_dataset_w_dataframe(dev_dataframe, args.data_dir, feature_extractor, args.max_length)
    test_dataset = create_dataset_w_dataframe(test_dataframe, args.data_dir, feature_extractor, args.max_length)

    # Create dataloader
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

    return train_loader, dev_loader, test_loader


def main(args):
    device = get_device(args.force_cpu)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        default=5,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )

    parser.add_argument(
        "--model_name_or_path",
        default="facebook/hubert-base-ls960",
        type=str,
        help=""
    )

    parser.add_argument(
        "--max_length",
        default=16000,
        type=int,
        help="max length for processing audio",
    )

    args = parser.parse_args()
    main(args)
