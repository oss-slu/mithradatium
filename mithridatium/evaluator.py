# mithridatium/evaluator.py
import torch

@torch.no_grad()
def extract_embeddings(model, dataloader, feature_module):
    """
    Collect penultimate features using a forward hook on `feature_module`
    (e.g., resnet.avgpool). Returns:
      embs:  [N, D] tensor
      labels:[N] tensor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    feats_list, labels_list = [], []

    def hook(_m, _inp, out):
        # avgpool output is [B, C, 1, 1]; flatten to [B, C]
        feats_list.append(out.detach().flatten(1).cpu())

    # Register the hook on the target layer
    handle = feature_module.register_forward_hook(lambda m, i, o: hook(m, i, o))
    try:
        for x, y in dataloader:
            x = x.to(device)
            _ = model(x)   # running forward triggers the hook
            labels_list.append(y) # keep labels to align with the embeddings
    finally:
        handle.remove()

    embs = torch.cat(feats_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return embs, labels
