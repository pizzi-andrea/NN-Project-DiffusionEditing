import clip
import torch

def directional_similarity(img1, 
                           img2,
                           text1:str, 
                           text2:str, 
                           model,
                           preprocess,
                           device = "cuda"):


    img1 = preprocess(img1).unsqueeze(0).to(device)
    img2 = preprocess(img2).unsqueeze(0).to(device)

    texts = clip.tokenize([text1, text2]).to(device)

    with torch.no_grad():
        img_features = model.encode_image(torch.cat([img1, img2]))
        text_features = model.encode_text(texts)

    img_features /= img_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    img_dir = (img_features[1] - img_features[0]).unsqueeze(0)
    text_dir = (text_features[1] - text_features[0]).unsqueeze(0)

    img_dir /= img_dir.norm(dim=-1, keepdim=True)
    text_dir /= text_dir.norm(dim=-1, keepdim=True)

    score = torch.cosine_similarity(img_dir, text_dir).item()


    return score