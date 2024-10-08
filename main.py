from train import train_model
import torch

if __name__ == "__main__":
    print('Training Model')
    model_type = #ResNet18 or ResNet18_SA
    tr_model, tstloss, trloss = train_model(model_type)
    torch.save(tr_model.state_dict(), '')
