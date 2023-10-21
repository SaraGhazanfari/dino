import torch
from advertorch.attacks import L2PGDAttack, LinfPGDAttack
import torch.nn as nn
from autoattack import AutoAttack


def generate_attack(attack, eps, model, x, target, loss=nn.CrossEntropyLoss()):
    attack_method, attack_norm = attack.split('-')
    if attack_method == 'AA':
        adversary = AutoAttack(model, norm=attack_norm, eps=eps, version='standard', device='cuda')
        adversary.attacks_to_run = ['apgd-ce']
        adv_image = adversary.run_standard_evaluation(x, target.long(), bs=x.shape[0])
    elif attack_method == 'PGD':
        if attack_norm == 'L2':
            adversary = L2PGDAttack(model, eps=eps, loss_fn=nn.MSELoss(), nb_iter=10, eps_iter=0.01, rand_init=True,
                                    clip_min=0., clip_max=1., targeted=False)
        else:
            adversary = LinfPGDAttack(model, eps=eps, loss_fn=nn.MSELoss(), nb_iter=50, eps_iter=0.03, rand_init=True,
                                      clip_min=0., clip_max=1., targeted=False)
        adv_image = adversary(x, target)
    return adv_image
