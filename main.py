from train import train_model
import torch

if __name__ == "__main__":
    print('Training Model')
    tr_model, tstloss, trloss = train_model()
    torch.save(tr_model.state_dict(), '')
