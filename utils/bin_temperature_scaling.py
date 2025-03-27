import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithBinwiseTemperature(nn.Module):
    def __init__(self, model, n_bins=10, device='cuda:5'):
        super().__init__()
        self.model = model
        self.n_bins = n_bins
        self.device = device
        self.temperatures = nn.Parameter(torch.ones(n_bins))  # bin별 temperature
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def __getattr__(self, name):
        """forward_features 등 내부 모델의 메서드에 접근 가능하도록 위임"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

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
        softmaxes = F.softmax(logits, dim=1)
        confidences, _ = torch.max(softmaxes, dim=1)
        scaled_logits = torch.zeros_like(logits)

        for i in range(self.n_bins):
            in_bin = (confidences > self.bin_lowers[i].item()) & (confidences <= self.bin_uppers[i].item())
            if in_bin.any():
                scaled_logits[in_bin] = logits[in_bin] / self.temperatures[i]

        return scaled_logits

    def set_temperature(self, valid_loader):
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss(n_bins=self.n_bins).to(self.device)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                if label.ndim > 1 and label.size(1) > 1:
                    label = torch.argmax(label, dim=1)
                if label.ndim > 1:
                    label = label.view(-1)
                logits = self.model.forward_features(input)[:, 0, :]
                logits = self.model.linear(logits)
                logits_list.append(logits.cpu())
                labels_list.append(label.cpu())

        logits = torch.cat(logits_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)

        before_nll = nll_criterion(logits, labels).item()
        before_ece = ece_criterion(logits, labels).item()
        print(f"Before BTS - NLL: {before_nll:.3f}, ECE: {before_ece:.3f}")

        optimizer = optim.LBFGS([self.temperatures], lr=0.01, max_iter=100)

        def eval():
            optimizer.zero_grad()
            scaled_logits = self.temperature_scale(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        after_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print(f"After BTS - NLL: {after_nll:.3f}, ECE: {after_ece:.3f}")
        print(f"Optimized bin-wise temperatures: {self.temperatures.data.cpu().numpy()}")

        return self

    def get_temperature(self):
        return self.temperatures


class _ECELoss(nn.Module):
    def __init__(self, n_bins=10):
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
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
