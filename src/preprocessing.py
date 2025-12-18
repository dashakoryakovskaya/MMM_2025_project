import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image


def _preprocess(df):
    df['activation_date'] = pd.to_datetime(df['activation_date'])

    df['day'] = df['activation_date'].dt.day
    df['month'] = df["activation_date"].dt.month
    df['year'] = df["activation_date"].dt.year
    df['weekday'] = df['activation_date'].dt.weekday
    df["dayofyear"] = df['activation_date'].dt.dayofyear
    df.drop(columns=['activation_date', 'item_id'], inplace=True)
    df['param_1'] = df['param_1'].fillna('')
    df['param_2'] = df['param_2'].fillna('')
    df['param_3'] = df['param_3'].fillna('')
    df['description'] = df['description'].fillna('')
    return df


def preprocess(df, image_dir):
    # text feature extractor
    feature_extractor_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", code_revision='da863dd04a4e5dce6814c6625adfba87b83838aa', trust_remote_code=True)
    feature_extractor_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", code_revision='da863dd04a4e5dce6814c6625adfba87b83838aa', trust_remote_code=True)

    # image feature extractor
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    image_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    df_preprocess = _preprocess(df)
    text = list(df_preprocess.apply(lambda item: '\n'.join([ item["title"], str(item["description"]), item["region"], item["city"], item["parent_category_name"], item["category_name"], ('' if item["param_1"] is None else str(item["param_1"])), ('' if item["param_2"] is None else str(item["param_2"])), ('' if item["param_3"] is None else str(item["param_3"]))]), axis=1).values)
    text_embedding = []
    for t in text:
        encoded_input = feature_extractor_tokenizer(t, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            features = feature_extractor_model(**encoded_input)[0][0]
        text_embedding.append(features)

    #user_type_dict = {'Private': 0, 'Company': 1, 'Shop': 2}
    #tabular = list(df_preprocess.apply(lambda item: torch.tensor([item["item_seq_number"], item["day"], item["month"], item["year"], item["weekday"], item["dayofyear"], user_type_dict[item["user_type"]], 0.0 if item["price"] is None else item["price"]]), axis=1).values)

    image_embedding = []
    for index, row in df.iterrows():
        res = torch.zeros(197, 768)
        if str(row['image']) != 'nan':
            image = Image.open(image_dir + '/' + str(row['image']) + '.jpg')
            inputs = image_processor(images=image, return_tensors="pt")
            outputs = image_model(**inputs)
            res = outputs.last_hidden_state.squeeze(dim=0)
        image_embedding.append(res)
    
    return text_embedding, image_embedding