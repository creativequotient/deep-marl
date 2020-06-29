import torch.nn.functional as F

from utils.common import onehot_from_logits


class ActionSelector(object):
    def __init__(self, act_shape_n, act_type, args, device):
        self.act_shape_n = act_shape_n
        self.act_type = act_type
        self.noise_n = []
        self.device = device
        for act_shape in self.act_shape_n:
            self.noise_n.append(lambda x: x if self.act_type == 'continuous' else None ) # TODO: add noise params

    def __call__(self, logits_n, explore, mask=None):
        if explore and self.act_type == 'continuous':
            logits_n = [noise(logits) for noise, logits in zip(self.noise_n, logits_n)]
            return logits_n
        elif self.act_type == 'continuous':
            return logits_n

        if mask is None:
            if explore:
                return [F.gumbel_softmax(logits, hard=True) for logits in logits_n]
            else:
                return [onehot_from_logits(logits, self.device) for logits in logits_n]
        else:
            raise NotImplementedError
