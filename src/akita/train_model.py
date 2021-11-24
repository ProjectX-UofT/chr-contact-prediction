import models
from pytorch_lightning import Trainer


class ModelTrainer:
    def __init__(self, learning_rate=0.05):
        self.learning_rate = learning_rate
        self.contact_predictor = models.ContactPredictor()
        self.lit_contact_predictor = models.LitContactPredictor(
            self.contact_predictor, self.learning_rate)
        self.model_trainer = None
        # TODO: add data module stuff when ready
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def train_model(self, auto_select_gpu=True, accelerator="gpu",
                    auto_lr_find=False, enable_checkpointing=True):
        self.model_trainer = Trainer(auto_select_gpu=auto_select_gpu,
                                     accelerator=accelerator,
                                     auto_lr_find=auto_lr_find,
                                     enable_checkpointing=enable_checkpointing)
        self.model_trainer.fit(self.lit_contact_predictor, self.train_loader)

    def validate_model(self):
        return self.model_trainer.validate(self.lit_contact_predictor,
                                           self.val_loader)

    def test_model(self):
        return self.model_trainer.test(self.lit_contact_predictor,
                                       self.test_loader)


if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.train_model()
