import hydra
from omegaconf import DictConfig

from models.grucls import GRUCLS


@hydra.main(config_path="../config", config_name="inference_config2.yaml")
def inference(cfg: DictConfig):
    models = [GRUCLS.load_from_checkpoint(x) for x in cfg.model.path_to_checkpoints]
    with open(cfg.log.file, "w", encoding="UTF-8") as file:
        while True:
            text = input("Введите сообщение: ")
            file.write(f"\n Введите сообщение: {text}")
            for model in models:
                out = model.generate_answer(text)
                print(f"Ответ {model}: {out}")
                file.write(f"\n Ответ {model}: {out}")
            print()


if __name__ == "__main__":
    inference()  # noqa pylint: disable=no-value-for-parameter
