import torch
import torch.nn.functional as F
import yaml
from models import  CLIPWrapper2

device = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    config_dir = 'models/configs/ViT.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)['ViT-L/14']

    model = CLIPWrapper2('ViT-L/14', config, 8, model_path='chek/epoch=31-step=5823.ckpt').to(device)
    model.eval()
    with torch.no_grad():
        image , text = []
        image_encoder = model.model.encode_image()
        text_encoder = model.model.encode_text()
        logits = F.normalize(image_encoder(image), dim=1) @ F.normalize(text_encoder(text),
                                                                            dim=1).t() * model.logit_scale.exp()
        image_logits= logits
        text_logits =logits.T
        print("hello world!")


if __name__ == '__main__':
    main()