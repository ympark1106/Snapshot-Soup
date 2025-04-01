import torch
from fvcore.nn import FlopCountAnalysis, parameter_count

def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """
    Calculate the FLOPs and number of parameters for a given model.
    
    Args:
        model (torch.nn.Module): The model for which to calculate FLOPs and params.
        input_size (tuple): The size of the input tensor (batch_size, channels, height, width).
    
    Returns:
        flops (float): Total FLOPs in GFLOPs.
        params (float): Total number of parameters in MParams.
    """
    # Generate a random input tensor with the given size
    inputs = torch.randn(input_size).to(next(model.parameters()).device)
    
        # Add a batch dimension if the input doesn't have one
    if inputs.dim() == 3:  # Check if there are only 3 dimensions (no batch dimension)
        inputs = inputs.unsqueeze(0)  # Add batch dimension: (1, 3, 224, 224)
    
    # Calculate FLOPs
    flops = FlopCountAnalysis(model, inputs)
    
    # Calculate the number of parameters
    params = parameter_count(model)
    
    # Convert FLOPs to GFLOPs and params to MParams
    total_flops = flops.total() / 1e9  # GFLOPs
    total_params = sum(params.values()) / 1e6  # MParams
    
    # Print the results
    print(f"FLOPs: {total_flops:.4f} GFLOPs")
    print(f"Number of parameters: {total_params:.4f} MParams")
    
    return total_flops, total_params
