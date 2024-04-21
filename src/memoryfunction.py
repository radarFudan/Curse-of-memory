import numpy as np
import torch


def memoryfunction(model, D, seed=233):
    """Given a seq2seq model: B * T * D -> B * T * D
        Evaluate the memory function defined in paper https://openreview.net/forum?id=yC2waD70Vj
    """
    torch.manual_seed(seed)

    B, T, D = 1, 1024, D
    x = torch.randn(B, D)[:,None,] * torch.ones(T)[None,:,None] # Heaviside inputs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        y, hidden = model(x)
    mf = torch.abs(y[:,1:,:] - y[:,:-1,:])
    
    if (D > 1):
        mf = torch.linalg.norm(mf[0], dim=-1)
    else:
        mf = torch.squeeze(mf)
    
    assert len(mf.shape) == 1, f"y should be 1D, got {mf.shape}"

    return mf.cpu().numpy()