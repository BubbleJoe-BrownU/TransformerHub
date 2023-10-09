import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, "../data/cifar10/")
from prepare_cifar10 import get_data_CIFAR
from modules import EncoderLayer, LearnablePositionEmbedding

    
class InputEmbedding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.patch_size = args.patch_size
        self.n_channels = args.n_channels
        self.latent_size = args.latent_size
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.batch_size = args.batch_size
        self.input_size = self.patch_size * self.patch_size * self.n_channels
        
        self.projection = nn.Linear(self.input_size, self.latent_size)
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)
        self.position = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)

    def forward(self, inputs):
        inputs = inputs.to(self.device)

        patchify = PatchExtractor(patch_size=self.patch_size)
        patches = patchify(inputs)

        linear_projection = self.projection(patches)
        b, n, _ = linear_projection.shape
        linear_projection = torch.cat([self.class_token, linear_projection], dim=1)
        pos_embedding = self.position[:, :n+1, :]
        embedding = linear_projection + pos_embedding

        return embedding


class MiniViT(nn.Module):
    def __init__(self, img_size, patch_size, embed_size, num_heads, num_layers, num_classes):
        super().__init__()
        assert img_size % patch_size == 0, f"image size ({img_size}) is not divisible by patch_size ({patch_size})"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size)**2


        self.projection = nn.Linear(patch_size**2*3, embed_size)
        self.cls_token = nn.Parameter(torch.zeros((1, 1, embed_size)))
        self.position_embedding = LearnablePositionEmbedding(self.num_patches+1, embed_size)
        
        self.encoder = nn.ModuleList([EncoderLayer(embed_size, num_heads, self.num_patches+1, dropout=0.3) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_size, 2*embed_size),
            nn.LeakyReLU(),
            nn.Linear(2*embed_size, num_classes)
        )

    def encoder_forward(self, inputs):
        batch_size = inputs.shape[0]
        patches = self.patchify(inputs)
        input_seq = self.projection(patches)
        # prepend cls_token to the sequence
        # and add position embedding
        input_seq = torch.cat([self.cls_token.repeat(batch_size, 1, 1), input_seq], dim=1)
        pos_embedding = self.position_embedding(torch.arange(self.num_patches+1, device=inputs.device).reshape(1, -1))
        embedding = input_seq + pos_embedding
        for layer in self.encoder:
            embedding = layer(embedding)
        return embedding

    def forward(self, inputs):
        embedding = self.encoder_forward(inputs)
        cls_embedding = embedding[:, 0, :]
        return self.mlp_head(cls_embedding)
        

    def patchify(self, inputs):
        """
        inputs: (N, 3, H, W)
        patches: (N, num_patches, patch_size**2 *3)
        """
        batch_size, channels, height, width = inputs.shape
        num_patches_h, num_patches_w = height // self.patch_size, width // self.patch_size
        patches = inputs.reshape(batch_size, channels, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
        patches = torch.einsum('nchpwq->nhwcpq', patches)
        patches = patches.reshape(batch_size, num_patches_h*num_patches_w, self.patch_size**2 *3)
        return patches

def generate_batches(batch_size=128, split="train", data_path="../data/cifar10"):
    images, labels, _ = get_data_CIFAR(split, data_path=data_path)
    len_dataset = len(images)
    # shuffle the dataset
    indices = torch.randperm(len_dataset)
    for i in range(0, len_dataset, batch_size):
        end = min(len_dataset, i+batch_size)
        yield images[indices[i:end]], labels[indices[i:end]]
    
def train(model, optimizer, max_epochs, batch_size, device):
    

    for epoch in range(max_epochs):
        dataloader = generate_batches(batch_size=batch_size, split="train")
        losses = []
        for inputs, targets in dataloader:
            inputs = torch.from_numpy(inputs).to(device)
            targets = torch.from_numpy(targets).to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        print(f"Training epoch: {epoch}, Training loss: {sum(losses)/len(losses)}")

        # evaluate the model every two epochs or at the last epoch
        if ((epoch+1) % 2 == 0) or (epoch+1 == max_epochs):
            dataloader = generate_batches(batch_size=batch_size, split="test")
            
            losses = []
            num_corrects = 0
            total_num = 10_000
            for inputs, targets in dataloader:
                inputs = torch.from_numpy(inputs).to(device)
                targets = torch.from_numpy(targets).to(device)
                with torch.no_grad():
                    logits = model(inputs)
                    loss = F.cross_entropy(logits, targets)
                num_corrects = (logits.argmax(-1) == targets).sum()
                losses.append(loss.item())
            print(f"Evaluation at epoch: {epoch}, Evaluation loss: {sum(losses)/len(losses)}, Evaluation accuracy: {num_corrects/total_num*100:.2f}%")
            
        

    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[dtype]
    to_compile = True if (torch.__version__.startswith("2") and device=="cuda") else False
    
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cuda.allow_tf32 = True # allow tf32 on cudnn
    print(device)
    model = MiniViT(32, 16, 768, 2, 6, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, optimizer, 5, 256, device)