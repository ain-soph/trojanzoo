# -*- coding: utf-8 -*-


from .latent_backdoor import Parser_Latent_Backdoor


class Parser_IMC_Latent(Parser_Latent_Backdoor):
    r"""IMC Latent Backdoor Attack Parser
    Attributes:
        name (str): ``'attack'``
        attack (str): ``'imc_latent'``
    """
    attack = 'imc_latent'

    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)

        parser.add_argument('--pgd_alpha', dest='pgd_alpha', type=float)
        parser.add_argument('--pgd_epsilon', dest='pgd_epsilon', type=float)
        parser.add_argument('--pgd_iteration', dest='pgd_iteration', type=int)
