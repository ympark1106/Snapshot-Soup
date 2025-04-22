import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.cuda.amp.autocast_mode import autocast
import matplotlib.pyplot as plt


def rein_forward(model, inputs):
    output = model.forward_features(inputs)[:, 0, :]
    output = model.linear(output)
    output = torch.softmax(output, dim=1)
    return output

def lora_forward(model, inputs):
    with autocast(enabled=True):
        features = model.forward_features(inputs)
        output = model.linear(features)
        output = torch.softmax(output, dim=1)
    return output

def adaptformer_forward(model, inputs):
    f = model.forward_features(inputs)
    outputs = model.linear(f)
    outputs = torch.softmax(outputs, dim=1) 
    return outputs

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, device = 'cuda:0'):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0) 
        # self.temperature = nn.Parameter(torch.ones(1) * 0.7)
        self.device = device
        
        
        
    def forward(self, input):
        # Use the model's forward_features and then apply temperature scaling
        if hasattr(self.model, 'forward_features'):
            logits = self.model.forward_features(input)[:, 0, :]
        else:
            logits = self.model(input)

        # Apply the linear layer if it exists in the model
        if hasattr(self.model, 'linear'):
            logits = self.model.linear(logits)
        else:
            raise AttributeError(f"The model {type(self.model).__name__} does not have a 'linear' layer.")

        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        # temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / self.temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, cross_validate = 'ece', args = None):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        # self.to(device)
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                if label.ndim > 1 and label.size(1) > 1:
                    label = torch.argmax(label, dim=1)
                if label.ndim > 1:
                    label = label.view(-1) 
                if args.type == 'rein':
                    logits = rein_forward(self.model, input)
                elif args.type == 'lora':
                    logits = lora_forward(self.model, input)
                elif args.type == 'adaptformer':
                    logits = adaptformer_forward(self.model, input)
                # print(logits.shape)                         # 임시 출력
                logits_list.append(logits.cpu())
                labels_list.append(label.cpu())
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
            

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=100)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            # print(f"[DEBUG] Current temperature: {self.temperature.item()}, Loss: {loss.item()}")
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        
        # 시각화 코드
        # plt.hist(F.softmax(logits, dim=1).cpu().detach().numpy().max(axis=1), bins=10, alpha=0.5, label="Before TS")
        # plt.hist(F.softmax(logits / self.temperature, dim=1).cpu().detach().numpy().max(axis=1), bins=10, alpha=0.5, label="After TS")
        # plt.legend()
        # plt.show()  

        return self
    
    def get_temperature(self):
        return self.temperature


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    
    
    
    