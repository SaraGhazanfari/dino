from advertorch.attacks import L2PGDAttack, LinfPGDAttack
import torch.nn as nn


def generate_attack(attack, eps, model, x, target, loss=nn.CrossEntropyLoss()):
    if attack == 'L2':
        adversary = L2PGDAttack(model, loss_fn=loss, eps=eps, nb_iter=1000, eps_iter=0.01, rand_init=True, clip_min=0.,
                                clip_max=1., targeted=False)
    else:
        adversary = LinfPGDAttack(model, loss_fn=loss, eps=eps, nb_iter=50, eps_iter=0.03, rand_init=True, clip_min=0.,
                                  clip_max=1., targeted=False)
    x = x.cuda()
    adv_image = adversary(x, target)
    return adv_image
