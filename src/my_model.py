import torch
from tqdm import tqdm
import logging, os
from datetime import datetime
from models.DEEP_MTLR import DEEP_MTLR
from torch.optim import Adam, AdamW, SGD
from my_data import *
from utils import *
import pickle

log_dir = 'TMSSv2_CV703/log/'  #change
out_dir = 'TMSSv2_CV703/weight/' #change
featureSize=24 #change

log_name = 'EHR_tmssv2_clip24_prompt3' #change
architecture = 'SWIN' #change

# log_name = 'tmss' #change
# architecture = 'UNETR' #tmss #change

if architecture=='SWIN':
    hparams = {'n_dense' : 2,   #Number of fc layers.
            'featureSize': featureSize,
            'hidden_size': 16*featureSize, # featureSize*16:swin, 768: vit
            'time_bins': 14,
            'dense_factor': 2, #Factor to multiply width of fc layer.
            'dropout': 0.2,
            'useBert': False}
else:
        hparams = {'n_dense' : 2,   #Number of fc layers.
            'hidden_size': 768, # featureSize*16:swin, 768: vit
            'time_bins': 14,
            'dense_factor': 2, #Factor to multiply width of fc layer.
            'dropout': 0.2,}

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f'_{log_name}'
logging.basicConfig(filename=os.path.join(log_dir, log_filename+'.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# train_loader,  val_loader = HECKTORDataModule().data()

filename = '/l/users/thao.nguyen/tmss_v4/src/train_loader.pkl'
with open(filename, 'rb') as file:
    train_loader = pickle.load(file)

filename = '/l/users/thao.nguyen/tmss_miccai/val_loader.pkl'
with open(filename, 'rb') as file:
    val_loader = pickle.load(file)

model = DEEP_MTLR(hparams,architecture)
model = model.to("cuda")
criterion = Dice_and_FocalLoss()
optimizer = make_optimizer(AdamW, model, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.01)
# scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)

Cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

best_loss=1e6
num_epochs = 50
for epoch in range(num_epochs):
    total = 0
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()
        # loss, _, _, _, _ = training_one_step(model = model, criterion = criterion, C1 = 100,loss_gamma=0.5,  batch=batch)
        loss, _ = training_one_step(model = model, criterion = criterion, C1 = 100,loss_gamma=0.5,  batch=batch)
        loss.backward()
        optimizer.step()

    if epoch < 31:
        StepLR.step()
    else:
        Cosine.step()

    # Evaluate the model
    total = 0
    val_loss=0
    val_mean_dice, val_ci_event = 0, 0
    val_loss, val_loss_mask, val_loss_mtlr = 0, 0, 0

    model.eval()
    for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):   
        # loss, loss_mtlr, loss_mask, mean_dice, ci_event = validation_one_step(model = model, criterion = criterion, C1 = 100,loss_gamma=0.5,  batch=batch)
        loss_mtlr, ci_event = validation_one_step(model = model, criterion = criterion, C1 = 100,loss_gamma=0.5,  batch=batch)
        if ci_event >0:
            total+=1
            val_loss +=loss
            # val_loss_mask += loss_mask.item()
            val_loss_mtlr += loss_mtlr.item()
            # val_mean_dice +=mean_dice
            val_ci_event += ci_event


    val_loss = val_loss/total
    val_loss_mtlr = val_loss_mtlr/total
    val_loss_mask=val_loss_mask/total
    val_mean_dice = val_mean_dice/total
    val_ci_event = val_ci_event /total

    # logging.info(f"Epoch {epoch + 1}/{num_epochs}, Validation - val_loss_mask: {val_loss_mask}, val_loss_mtlr: {val_loss_mtlr},val_mean_dice: {val_mean_dice}, val_ci_event: {val_ci_event}")
    logging.info(f"Epoch {epoch + 1}/{num_epochs}, Validation - val_loss_mtlr: {val_loss_mtlr}, val_ci_event: {val_ci_event}")

    if val_loss < best_loss:
        best_loss=val_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'model': model,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]['lr'],
                    }, out_dir+log_filename+'.pth') #Change
        print(f"Best Loss: {best_loss}")
