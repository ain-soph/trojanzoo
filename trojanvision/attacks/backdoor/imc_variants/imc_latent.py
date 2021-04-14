#!/usr/bin/env python3

from trojanvision.attacks.backdoor.latent_backdoor import LatentBackdoor


class IMC_Latent(LatentBackdoor):
    name: str = 'imc_latent'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.mark.random_pos:
            raise Exception('IMC requires "random pos" to be False to train mark.')

    def attack(self, **kwargs):
        print('Sample Data')
        self.data = self.sample_data()
        print('Retrain')
        return super(LatentBackdoor, self).attack(epoch_fn=self.epoch_fn, **kwargs)

    def epoch_fn(self, **kwargs):
        self.avg_target_feats = self.get_avg_target_feats(self.data)
        self.preprocess_mark(data=self.data)
