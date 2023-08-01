import hydra
from omegaconf import DictConfig

from grulm import GRULM


@hydra.main(config_path="config", config_name="inference_config.yaml")
def inference(cfg: DictConfig):
    model = GRULM.load_from_checkpoint(cfg.model.path_to_checkpoint)
    with open(cfg.log.file, "w", encoding="UTF-8") as file:
        while True:
            text = input("Введите сообщение: ")
            out = model.generate_sequence(text, cfg.model.max_tokens)
            print(f"Ответ: {out}")

            file.write(f"\n Введите сообщение: {text} \n Ответ: {out}")


if __name__ == "__main__":
    inference()  # noqa pylint: disable=no-value-for-parameter
