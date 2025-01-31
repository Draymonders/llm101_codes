import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, AdamW
from dataset import NewsDataset
from model import NewsClassifier
import torch.nn as nn
from tqdm import tqdm
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

"""    
Toutiao News Classification

Description:
- News text multi classification (15 classes) using BERT model
- Model training, evaluation and text representation visualization
- Supports both t-SNE and UMAP visualization methods

# transformers: 
#     - BertTokenizer: tokenize text
#     - BertModel: pre-trained model

Usage:
python main.py [--parameter value]

Main Parameters:
--model: Pre-trained model path
--data_path: Dataset path
--batch_size: Batch size (default: 128)
--epochs: Training epochs (default: 5)
--learning_rate: Learning rate (default: 2e-5)
--max_len: Maximum text length (default: 64)

TODO:
- add early_stop
- add logging, e.g. wandb/loguru
- add learning rate scheduler

"""


def set_seed(seed=42):
    """Set random seed to ensure reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(
        description="Toutiao News Classification Training Parameters"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/root/.cache/modelscope/hub/damo/nlp_bert_backbone_base_std",
        help="Pre-trained model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="toutiao_cat_data.txt",
        help="data path",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Whether to use toy data to test code correctness",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=5, help="training epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="learning rate"
    )
    parser.add_argument(
        "--max_len", type=int, default=64, help="text max sequence length"
    )
    parser.add_argument("--n_classes", type=int, default=17, help="number of classes")
    return parser.parse_args()


def visualize_embeddings(model, test_loader, device, method="tsne"):
    """
    Extract test data embeddings and visualize

    Args:
        model: trained model
        test_loader: test data loader
        device: computing device
        method: dimensionality reduction method, 'tsne' or 'umap'
    """
    model.eval()
    embeddings = []
    labels = []

    # collect all test data embeddings and labels
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="extracting embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["label"].cpu().numpy()

            # get BERT last layer [CLS] embeddings as text representation
            last_hidden_states = model.get_embeddings(input_ids, attention_mask)
            embeddings.append(last_hidden_states.cpu().numpy())
            labels.extend(batch_labels)

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)

    # [CLS] embeddings dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        try:
            import umap

            reducer = umap.UMAP(random_state=42)
        except ImportError:
            print("UMAP is not installed, using t-SNE instead")
            reducer = TSNE(n_components=2, random_state=42)

    embeddings_2d = reducer.fit_transform(embeddings)

    # visualization
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab20", alpha=0.6
    )
    plt.colorbar(scatter)
    plt.title(f"Text embeddings {method.upper()} visualization")
    plt.savefig(f"text_embeddings_{method}.png")
    plt.close()


def train_epoch(model, data_loader, optimizer, criterion, device):
    """
    Train one epoch

    Args:
        model: model instance
        data_loader: training data loader
        optimizer: optimizer
        criterion: loss function
        device: computing device (CPU/GPU)

    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        # print(f"loss: {loss.item()}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(data_loader), correct / total


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model performance

    Args:
        model: model instance
        data_loader: validation data loader
        criterion: loss function
        device: computing device (CPU/GPU)

    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(data_loader), correct / total


def dataloader(args):
    tokenizer = BertTokenizer.from_pretrained(args.model)
    dataset = NewsDataset(args.data_path, tokenizer, args.max_len, args.toy)
    print("dataset done")

    # split training set, validation set and test set (ratio 0.7:0.15:0.15)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    return train_loader, val_loader, test_loader

def train():
    args = get_args()
    set_seed(args.seed)
    # load data
    train_loader, val_loader, test_loader = dataloader(args)

    # create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NewsClassifier(args.n_classes, args.model).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model_state = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        print("-" * 60)

    # load best model for testing
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # save best model
    torch.save(best_model_state, "best_news_classifier.pth")

def visualization():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load data
    train_loader, val_loader, test_loader = dataloader(args)
    model = NewsClassifier(args.n_classes, args.model).to(device)
    model.load_state_dict(torch.load("best_news_classifier.pth"))

    # text embeddings visualization
    print("\nVisualizing analysis...")
    # visualize_embeddings(model, test_loader, device, method="tsne")
    visualize_embeddings(model, test_loader, device, method="umap")


if __name__ == "__main__":
    # python main.py --model=bert-base-chinese --batch_size=64 --learning_rate=0.01 --toy
    train()
    visualization()
