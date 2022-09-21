import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from data import BrainDataset, define_transforms, get_dataframe
from model import ResNeXtUNet
from utils import bce_dice_loss, compute_iou, dice_coef_loss, dice_coef_metric

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ROOT_PATH = './data/raw/lgg-mri-segmentation/kaggle_3m/'

df = get_dataframe(ROOT_PATH)

train_df, val_df = train_test_split(df, stratify=df['diagnosis'], test_size=0.1)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df, test_df = train_test_split(train_df, stratify=train_df['diagnosis'], test_size=0.15)
train_df = train_df.reset_index(drop=True)

print(df.head())
train_transform, val_transform, test_transform = define_transforms()

train_dataset = BrainDataset(train_df, train_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

val_dataset = BrainDataset(val_df, val_transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=2)

test_dataset = BrainDataset(test_df, test_transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)



def train_model(train_loader, val_loader, loss_func, optimizer, scheduler, num_epochs):
    loss_history = []
    train_history = []
    val_history = []
    
    for epoch in range(num_epochs):
        model.train()
        
        losses = []
        train_iou = []
        
        for i, (image, mask) in enumerate(tqdm(train_loader)):
            image = image.to(device)
            mask = mask.to(device)
            outputs = model(image)
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0            
            
            train_dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())
            loss = loss_func(outputs, mask)
            losses.append(loss.item())
            train_iou.append(train_dice)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
        val_mean_iou = compute_iou(model, device, val_loader)
        scheduler.step(val_mean_iou)
        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_iou)
        
        print('Epoch : {}/{}'.format(epoch+1, num_epochs))
        print('loss: {:.3f} - dice_coef: {:.3f} - val_dice_coef: {:.3f}'.format(np.array(losses).mean(),
                                                                               np.array(train_iou).mean(),
                                                                               val_mean_iou))
    return loss_history, train_history, val_history


if __name__ == "__main__":
    model = ResNeXtUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    num_epochs = 6
    loss_history, train_history, val_history = train_model(train_dataloader, val_dataloader, dice_coef_loss, optimizer, scheduler, num_epochs)
    torch.save(model, 'model.pth')

