import numpy as np
import torch as T


def onehot_from_logits(logits):
    logits = logits.detach().cpu().numpy()
    output = np.zeros(logits.shape)
    for idx, entry in enumerate(logits):
        output[idx][np.argmax(entry)] = 1.0
    return T.tensor(output, dtype=T.double, device=T.device('cuda' if T.cuda.is_available() else 'cpu'))
