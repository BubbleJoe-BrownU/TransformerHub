
import torch
import torch.nn as nn

import torchvision

from modules import EncoderLayer, LearnablePositionEmbedding


class PatchExtractor(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, inputs):
        # expected input shape [batch_size, channels, height, width]
        batch_size, channels, height, width = inputs.shape

        assert height % self.patch_size == 0 and width % self.patch_size == 0, \
            f"Input height ({height}) and width ({width}) must be divisible by patch size ({self.patch_size})"
        
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # patches = inputs.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0, 2, 3, 1, 4, 5).contiguous().view(batch_size, num_patches, -1)
        patches = inputs.reshape(batch_size, channels, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
        patches = torch.einsum('nchpwq->nhwcpq', patches).reshape(batch_size, num_patches, -1)
        # the returning patches has shape (batch_size, 14*14, 16*16*3)
        return patches
    
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
    def __init__(self, img_size, patch_size, embed_size, num_heads, num_layers):
        super().__init__()
        assert img_size % patch_size == 0, f"image size ({img_size}) is not divisible by patch_size ({patch_size})"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size)**2


        self.projection = nn.Linear(patch_size**2*3, embed_size)
        self.cls_token = nn.Parameter(torch.zeros((1, 1, embed_size)))
        self.position_embedding = LearnablePositionEmbedding(self.num_patches+1, embed_size)
        
        self.encoder = nn.ModuleList([EncoderLayer(embed_size, num_heads, self.num_patches+1, dropout=0.3) for _ in range(num_layers)])

    def forward(self, inputs):
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


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    inputs = torch.randn(10, 3, 224, 224).to(device)
    vit = MiniViT(224, 16, 768, 2, 6).to(device)
    output = vit(inputs)
    print(output.shape)