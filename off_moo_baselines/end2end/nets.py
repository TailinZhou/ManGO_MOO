import torch.nn as nn
import torch 
import os 
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
# activate_functions = [nn.LeakyReLU(), nn.LeakyReLU()]
 

class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x

activate_functions = [Swish(), Swish()]


class End2EndMultiHeadModel(pl.LightningModule):
    def __init__(self, n_dim, n_obj, hidden_size,
                 save_path=None):
        super(End2EndMultiHeadModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj

        layers = []
        layers.append(nn.Linear(n_dim, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        lastlayers = []
        for i in range(n_obj):
            lastlayers.append(nn.Linear(hidden_size[len(hidden_size)-1], 1))

        self.layers = nn.Sequential(*layers)
        self.lastlayers = nn.ModuleList(lastlayers)
        self.hidden_size = hidden_size
        self.save_path = save_path
    
    def forward(self, x):

        for i in range(len(self.hidden_size)):
            x = self.layers[i](x)
            x = activate_functions[i](x)
        out = []
        for i in range(self.n_obj):
            out_i = self.lastlayers[i](x)
            out.append(out_i)
        # x = self.layers[len(self.hidden_size)](x)
        # out = x
        out = torch.cat(out, dim=1)

        return out



    def set_kwargs(self, device=None, dtype=None):
        self.to(device=device, dtype=dtype)
    
    def check_model_path_exist(self, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        return os.path.exists(save_path)     
    
    def save(self, val_mse=None, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
            
        from off_moo_baselines.data import tkwargs
        
        self = self.to('cpu')
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if val_mse is not None:
            checkpoint["valid_mse"] = val_mse
        
        torch.save(checkpoint, save_path)
        self = self.to(**tkwargs)
    
    def load(self, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        
        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint["model_state_dict"])
        valid_mse = checkpoint["valid_mse"]
        print(f"Successfully load trained model from {save_path} " 
                f"with valid MSE = {valid_mse}")


class End2EndModel(pl.LightningModule):
    def __init__(self, n_dim, n_obj, hidden_size,
                 save_path=None):
        super(End2EndModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj

        layers = []
        layers.append(nn.Linear(n_dim, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        layers.append(nn.Linear(hidden_size[len(hidden_size)-1], n_obj))

        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_size
        
        self.save_path = save_path
    
    def forward(self, x):

        for i in range(len(self.hidden_size)):
            x = self.layers[i](x)
            x = activate_functions[i](x)
        
        x = self.layers[len(self.hidden_size)](x)
        out = x

        return out



    def set_kwargs(self, device=None, dtype=None):
        self.to(device=device, dtype=dtype)
    
    def check_model_path_exist(self, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        return os.path.exists(save_path)     
    
    def save(self, val_mse=None, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
            
        from off_moo_baselines.data import tkwargs
        
        self = self.to('cpu')
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if val_mse is not None:
            checkpoint["valid_mse"] = val_mse
        
        torch.save(checkpoint, save_path)
        self = self.to(**tkwargs)
    
    def load(self, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        
        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint["model_state_dict"])
        valid_mse = checkpoint["valid_mse"]
        print(f"Successfully load trained model from {save_path} " 
                f"with valid MSE = {valid_mse}")



class End2EndNonLinearModel(pl.LightningModule):
    def __init__(self, n_dim=None, n_obj=None, hidden_size=None,
                 save_path=None):
        super(End2EndNonLinearModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.save_path = save_path

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

    
    def training_step(self, batch, batch_idx, log_prefix="train"):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, log_prefix="val")
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def set_kwargs(self, device=None, dtype=None):
        self.to(device=device, dtype=dtype)
    
    def check_model_path_exist(self, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        return os.path.exists(save_path)     
    
    def save(self, val_mse=None, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
            
        from off_moo_baselines.data import tkwargs
        
        self = self.to('cpu')
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if val_mse is not None:
            checkpoint["valid_mse"] = val_mse
        
        torch.save(checkpoint, save_path)
        self = self.to(**tkwargs)
    
    def load(self, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        
        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint["model_state_dict"])
        valid_mse = checkpoint["valid_mse"]
        print(f"Successfully load trained model from {save_path} " 
                f"with valid MSE = {valid_mse}")
    


 
class End2EndClassificationModel(pl.LightningModule):
    def __init__(self, n_dim, n_obj, hidden_size,
                 save_path=None):
        super(End2EndClassificationModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj

        layers = []
        layers.append(nn.Linear(n_dim+n_obj, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        layers.append(nn.Linear(hidden_size[len(hidden_size)-1], 2))

        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_size
        
        self.save_path = save_path
    
    def forward(self, x ):
        # h = torch.cat([x, y], dim=1)
        h = x
        for i in range(len(self.hidden_size)):
            h = self.layers[i](h)
            h = activate_functions[i](h)
        
        out = self.layers[len(self.hidden_size)](h)

        return out
    
    def training_step(self, batch, batch_idx, log_prefix="train"):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long())
        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, log_prefix="val")
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def set_kwargs(self, device=None, dtype=None):
        self.to(device=device, dtype=dtype)
    
    def check_model_path_exist(self, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        return os.path.exists(save_path)     
    
    def save(self, val_mse=None, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
            
        from off_moo_baselines.data import tkwargs
        
        self = self.to('cpu')
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if val_mse is not None:
            checkpoint["valid_mse"] = val_mse
        
        torch.save(checkpoint, save_path)
        self = self.to(**tkwargs)
    
    def load(self, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        
        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint["model_state_dict"])
        valid_mse = checkpoint["valid_mse"]
        print(f"Successfully load trained model from {save_path} " 
                f"with valid MSE = {valid_mse}")
    
