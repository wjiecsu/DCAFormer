import torch
import torch.nn as nn
import numpy  as np
from   torch.utils.data import Dataset
import torch.optim as optim
from   tqdm import tqdm

def train_Transformer(model, train_loader, test_loader, max_epochs, path_models,Formername, device):
    # 输入输出
    # 1. 数据的特征 feature_num
    # 2. 数据的长度 seq_len
    # 3. 批训练个数 BATCH_SIZE
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    val_loss = []
    train_loss = []
    best_test_loss = np.inf
    for epoch in tqdm(range(max_epochs)):
        model.train()
        train_epoch_loss = []
        for index, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.type(torch.FloatTensor).to(device)
            targets = targets.type(torch.FloatTensor).to(device)
            ypred = model(inputs)
            loss = criterion(targets,ypred)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
        train_loss.append(np.mean(train_epoch_loss))
        _, _, val_epoch_loss = test_main(model, test_loader, device)
        val_loss.append(val_epoch_loss)
        print("epoch:", epoch, "train_epoch_loss:", np.mean(train_epoch_loss), "val_epoch_loss:", val_epoch_loss)
        # 保存下来最好的模型：
        if val_epoch_loss < best_test_loss:
            best_test_loss = val_epoch_loss
            best_model = model
            print("best_test_loss  -------------------------------------------------", best_test_loss)
            torch.save(best_model.state_dict(),path_models+'best_'+Formername+'.pth')
    val_loss = np.array(val_loss)
    train_loss = np.array(train_loss)
    return val_loss, train_loss
 
def test_main(model, test_loader, device):
    y_pred = []
    y_true = []
    val_epoch_loss = []
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for index, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.type(torch.FloatTensor).to(device)
            targets = targets.type(torch.FloatTensor).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            y_pred.append(outputs)
            y_true.append(targets)
            val_epoch_loss.append(loss.item())
        epoch_loss = np.mean(val_epoch_loss)
        y_pred = torch.cat(y_pred, 0).cpu().detach().numpy()
        y_true = torch.cat(y_true, 0).cpu().detach().numpy()
    return y_true, y_pred, epoch_loss