import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


LEARNING_RATE=1e-4
NUM_CLASSES = 10 #10 classes to classify
PATCH_SIZE=4 
IMG_SIZE=28
IN_CHANNELS=1
NO_OF_HEADS=8
DROPOUT=0.001
HIDDEN_DIM=768 
ADAM_WEIGHT_DECAY=0.1
ADAM_BETAS = (0.9, 0.999)
ACTIVATION="gelu"
NUM_ENCODERS=4
EMBED_DIM = (PATCH_SIZE**2) * IN_CHANNELS # 4^2*1=16
NUM_PATCHES = (IMG_SIZE//PATCH_SIZE)**2 # 28/4=7; 7^2=49
BATCH_SIZE=512
TOTAL_EPOCHS=40
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        # (b,n_channel,h,w) #(512,n_channel,28,28)
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size), nn.Flatten(2)
        )
        # op size = (input-kernel)/stride +1 => ((28-4)\2)+1=7 --> (512,1,7,7)
        #nn.Flatten(2) -> flattens last 2 dims. from B,1,H,W -> B,E,H*W
        # before flatten - (512, 1, 7,7) after --> (512,16,49)
        self.cls_token = nn.Parameter(torch.randn(size=(1, in_channels, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # (b,n_channel,height, width) ip
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # (1, 1, embed_dim) -> (b,1,16)
        # x -> (b,c,h,w) -> (b,embed_dim,op_size)

        # (512, 1, 28, 28) -> (512,16, 7*7)
        x=self.patcher(x)
        # exchanges 1,2 idx -> (512,49,embed_dim)
        x=x.permute(0,2,1)
        
        #cls token + x-> (1,1,embed_dim) + (512,49,embed_dim) -> (512, 50, embed_dim)
        x=torch.cat(
            [cls_token, x], dim=1
        )
        x+=self.position_embeddings
        x= self.dropout(x)

        return x

embd_model = PatchEmbedding(EMBED_DIM, PATCH_SIZE, NUM_PATCHES, DROPOUT, IN_CHANNELS).to(device)
test_ip=torch.randn(512, 1, 28, 28) # b, c, height, width
# [512, 50, 16]
assert embd_model(test_ip).shape == torch.Size([BATCH_SIZE, NUM_PATCHES+1, EMBED_DIM])


class VIT(nn.Module):
    def __init__(self, num_patches, num_classes, patch_size, embed_dim, num_encoders, num_heads, dropout, activation, in_channels):
        super().__init__()
        self.embedding_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,nhead=num_heads, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, num_encoders)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), 
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # (ip -> b,channel, h,w)
        x=self.embedding_block(x) # (512, total_patches+1, embed_dim)
        x=self.encoder_block(x) # (512, total_patches+1, embed_dim)
        x=self.mlp_head(x[:,0, :]) # (512, embed_dim) --> [512,10]
        return x #(512,10)
    
model = VIT(NUM_PATCHES, NUM_CLASSES, PATCH_SIZE, EMBED_DIM, NUM_ENCODERS, NUM_ENCODERS, DROPOUT, ACTIVATION, IN_CHANNELS).to(device)
test_ip=torch.randn(512, 1,28,28)
x=model(test_ip)
assert x.shape == torch.Size([BATCH_SIZE, NUM_CLASSES])

# transform to tensor and normalize
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))
    ]
)

train_dataset=datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset=datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=64, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
optimiser=torch.optim.Adam(model.parameters(), LEARNING_RATE, ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)

for epoch in range(TOTAL_EPOCHS):
    print('-'*100)
    print('Training on ', device)
    print(f'Epoch: {epoch}')
    
    model.train()
    running_loss =0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs=model(images)
        loss=loss_fn(outputs, labels)
        loss.backward()
        optimiser.step()
        running_loss+=loss.item()
    epoch_loss = running_loss / len(train_dataset)
    print('Epoch Loss: ', epoch_loss)
