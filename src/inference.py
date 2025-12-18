import torch

from model import MultiModalTransformer
from preprocessing import preprocess


def inference(df, path_to_checkpoint):
    text_embeddings, image_embeddings = preprocess(df)
    model = MultiModalTransformer(first_dim=1024, second_dim=768)
    checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    result = []
    for i in range(df.shape[0]):
        result.append(float(model(text_embeddings[i].unsqueeze(0), image_embeddings[i].unsqueeze(0))))

    return result
