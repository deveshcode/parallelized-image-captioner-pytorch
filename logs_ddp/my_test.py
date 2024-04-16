import pickle
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import models
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import Dataset
from torch.cuda.amp import GradScaler, autocast
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import json 
from collections import Counter
from tqdm import tqdm
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
import time
import multiprocessing
import matplotlib.pyplot as plt
from torchvision.datasets import CocoDetection
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torchvision import models
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

annotations = "/scratch/surve.de/caption_data_75k.json"
image_dir = "/scratch/surve.de/devesh/COCO"

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # Add special tokens
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)



def build_vocabulary(data, threshold):
    """Builds a vocabulary from the dataset captions.

    Args:
        data (list): List of dictionaries with 'image' and 'caption'.
        threshold (int): Minimum word frequency to be included in the vocabulary.

    Returns:
        Vocabulary: The constructed vocabulary.
    """
    counter = Counter()
    for item in data:
        tokens = word_tokenize(item['caption'].lower())
        counter.update(tokens)

    # Create a vocabulary
    vocab = Vocabulary()

    # Add words to the vocabulary if they meet the frequency threshold
    for word, count in counter.items():
        if count >= threshold:
            vocab.add_word(word)

    return vocab


with open(annotations, 'r') as f:
    json_data = json.load(f)

ann = []

for filename, data in json_data.items():
    image_id = data['image_id']
    caption = data['caption']
    ann.append({"image": filename, "caption": caption})

vocab = build_vocabulary(ann, 1)


class ImageCaptionDataset(Dataset):
    def __init__(self, img_dir, data, vocab, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            data (list): List of dictionaries with 'image' and 'caption'.
            vocab (Vocabulary): Vocabulary instance.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.data = data
        self.vocab = vocab
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.img_dir, self.data[idx]['image'])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Convert caption to list of word indices
        caption = self.data[idx]['caption']
        tokens = word_tokenize(caption.lower())
        caption_indices = [self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')]
        
        return image, torch.tensor(caption_indices)
    

class Encoder_SingleGPU(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder_SingleGPU, self).__init__()
        resnet = models.resnet50(pretrained=True)
        
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)  # Flatten the features
#         print(f"Encoder output shape: {features.shape}")  # Add this line
        return features


class Decoder_SingleGPU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder_SingleGPU, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, features, captions):
        features = features.to("cuda:1")
        embeddings = self.embed(captions)
        # Assuming you've adjusted the use of features as per your chosen strategy
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    
def setup_logging(rank):
    log_directory = "/home/surve.de/ondemand/logs"
    os.makedirs(log_directory, exist_ok=True)
    log_file = f"{log_directory}/log_6apr_{rank}.txt"
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s %(levelname)s %(message)s', filemode='w')


def collate_fn(batch):
    """
    Custom collate function to pad captions to the same length in each batch.
    
    Args:
        batch: List of tuples (image, caption), where caption is a tensor of word indices.
    
    Returns:
        images: Tensor of all images in the batch.
        captions: Tensor of all padded captions in the batch.
    """
    # Separate the images and captions
    images, captions = zip(*batch)

    # Convert images to a tensor
    images = torch.stack(images, 0)

    # Pad the captions to the maximum length caption in the batch
    captions = pad_sequence(captions, batch_first=True, padding_value=0) # Assuming 0 is the index for <pad>

    return images, captions


def train(rank, world_size, name_exp, mp_flag, batch_size, msg):
    setup_logging(rank)
    logging.info(f"Initializing training for rank {rank} out of {world_size} processes")
    
    # Initialize the distributed environment.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    logging.info(f"Setting master address and port")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    logging.info(f"Distributed process group initialized")

    # Assuming you have defined image_dir, ann, and vocab somewhere

    # Assuming you have defined ImageCaptionDataset, Encoder_SingleGPU, Decoder_SingleGPU, collate_fn

    dataset = ImageCaptionDataset(img_dir=image_dir, data=ann, vocab=vocab)
    logging.info(f"Dataset initialized")
    logging.info(f"Dataset Size :{len(dataset)}")
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    logging.info(f"Sampler initialized")

    device = torch.device(f'cuda:{rank}')
    logging.info(f"Device set to {device}")

    vocab_size = len(vocab)
    embed_size = 256
    hidden_size = 512
    num_layers = 1

    # Model setup
    encoder = Encoder_SingleGPU().to(device)
    decoder = Decoder_SingleGPU(embed_size, hidden_size, vocab_size, num_layers).to(device)
    logging.info(f"Models initialized on device")

    encoder = DDP(encoder, device_ids=[rank])
    decoder = DDP(decoder, device_ids=[rank])
    logging.info(f"Models wrapped with DistributedDataParallel")

    criterion = nn.CrossEntropyLoss().to(device)
    logging.info(f"Criterion set to CrossEntropyLoss")

    params = list(decoder.parameters()) + list(encoder.module.resnet.parameters())
    optimizer = optim.Adam(params, lr=0.001)
    logging.info(f"Optimizer set to Adam")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn, sampler=sampler)
    logging.info(f"Dataloader initialized")

    num_epochs = 5
    scaler = GradScaler(enabled=mp_flag)
    metrics = {
        'time': [],
        'loss': []
    }

    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        sampler.set_epoch(epoch)
        logging.info(f"Starting epoch {epoch + 1}")
        total_loss = 0
        
        for images, captions in data_loader:
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()

            with autocast(enabled=mp_flag):
                features = encoder(images)
                outputs = decoder(features, captions[:, :-1])
                loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        metrics['time'].append(epoch_duration)
        metrics['loss'].append(total_loss / len(data_loader))
        logging.info(f"Epoch {epoch+1}, Loss: {metrics['loss'][-1]}, Time: {metrics['time'][-1]} sec")
    
    if rank == 0:
        logging.info(f"{metrics}")
        exp = {f"{msg}" : metrics}
        print(exp)
        with open(f'{name_exp}.pkl', 'wb') as fp:
            pickle.dump(exp, fp)

    # Cleanup
    dist.destroy_process_group()
    logging.info(f"Training completed and process group destroyed")

    
