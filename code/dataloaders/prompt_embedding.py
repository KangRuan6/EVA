import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from PIL import Image
import random
import numpy as np


# 使用bert对文本进行embedding
def text_embedding(bert_tokenizer, bert_model, sentence, length):
    encoded_input = bert_tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**encoded_input)
    last_hidden_states = outputs[0]
    embedding = last_hidden_states[0].mean(dim=0)
    sentence_embedding = torch.nn.Linear(768, length)(embedding)
    return sentence_embedding


# 使用VIT对图片进行embedding
def img_embedding(VIT_processor, VIT_model, image, length):
    inputs = VIT_processor(images=image, return_tensors="pt")
    outputs = VIT_model(**inputs)
    feature = torch.nn.Linear(1000, length)(outputs.logits.squeeze(0))
    return feature


# length实验1：使用等比例的label:mask:caption=1:1:2
def concat_fusion(vit_processor, vit_model, bert_tokenizer, bert_model,
                  mask, label, caption,
                  mask_token, label_token, caption_token):
    # mask caption label embedding
    mask_embedding = img_embedding(vit_processor, vit_model, mask, mask_token)
    label_embedding = text_embedding(bert_tokenizer, bert_model, label, label_token)
    caption_embedding = text_embedding(bert_tokenizer, bert_model, caption, caption_token)
    # 拼接mask,label,caption的embedding
    concat_embedding = torch.cat((caption_embedding, label_embedding, mask_embedding), dim=0)
    return concat_embedding


# length实验2,3：使用gamma分布确定label:mask:caption的值 0.1,10
def concat_fusion_beta(vit_processor, vit_model, bert_tokenizer, bert_model,
                       mask, label, caption,
                       alpha, beta, length):
    beta_number = np.random.beta(alpha, beta)
    label_token = int(length * beta_number // 2)
    mask_token = int(length * beta_number // 2)
    caption_token = length - label_token - mask_token
    print(label_token, mask_token, caption_token)
    prompt = concat_fusion(vit_processor=vit_processor, vit_model=vit_model,
                           bert_tokenizer=bert_tokenizer, bert_model=bert_model,
                           mask=mask, label=label, caption=caption,
                           label_token=label_token, mask_token=mask_token, caption_token=caption_token)
    return prompt


# length 实验5：使用prompt句式作为embedding
def concat_fusion_prompt(vit_processor, vit_model, bert_tokenizer, bert_model,
                         mask, label, caption,
                         mask_token, prompt_token):
    mask_embedding = img_embedding(vit_processor, vit_model, mask, mask_token)
    caption_prompt = caption
    label_prompt = 'The label is ' + label + '.'
    mask_prompt = 'The mask is ' + str(mask_embedding.tolist()) + '.'
    strings = [caption_prompt, label_prompt, mask_prompt]
    random_strings = random.sample(strings, len(strings))
    random_prompt = ''.join(random_strings)
    prompt_embedding = text_embedding(bert_tokenizer, bert_model, random_prompt, prompt_token)
    print(random_prompt)
    return prompt_embedding


if __name__ == '__main__':
    img = Image.open(
        '/home/yqx/yqx_softlink/data/UCF101/UCF101_frames_16_mask_result/val/ApplyLipstick/v_ApplyLipstick_g02_c01/000019.jpg')
    label = 'flower'
    caption = 'There is a red flower in the picture.'
    mask = transforms.ToTensor()(img)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    VIT_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    VIT_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    Multimodal_fusion_mode = 'concat-fusion-beta-10'  # 生成prompt的模式
    '''
    ['concat-fusion-equal','concat-fusion-beta-0.1','concat-fusion-beta-10','concat-fusion-prompt']
    '''
    if Multimodal_fusion_mode == 'concat-fusion-equal':
        length = 64
        prompt = concat_fusion(vit_processor=VIT_processor, vit_model=VIT_model,
                               bert_tokenizer=bert_tokenizer, bert_model=bert_model,
                               mask=mask, label=label, caption=caption,
                               label_token=length // 4, mask_token=length // 4, caption_token=length // 2)
        print(prompt.shape)

    if Multimodal_fusion_mode == 'concat-fusion-beta-0.1':
        length = 64
        alpha = 0.1
        beta = 0.1
        prompt = concat_fusion_beta(vit_processor=VIT_processor, vit_model=VIT_model,
                                    bert_tokenizer=bert_tokenizer, bert_model=bert_model,
                                    mask=mask, label=label, caption=caption,
                                    alpha=alpha, beta=beta, length=length)
        print(prompt.shape)

    if Multimodal_fusion_mode == 'concat-fusion-beta-10':
        length = 64
        alpha = 10
        beta = 10
        prompt = concat_fusion_beta(vit_processor=VIT_processor, vit_model=VIT_model,
                                    bert_tokenizer=bert_tokenizer, bert_model=bert_model,
                                    mask=mask, label=label, caption=caption,
                                    alpha=alpha, beta=beta, length=length)
        print(prompt.shape)

    if Multimodal_fusion_mode == 'concat-fusion-prompt':
        mask_token = 16
        prompt_token = 64
        prompt = concat_fusion_prompt(vit_processor=VIT_processor, vit_model=VIT_model,
                                      bert_tokenizer=bert_tokenizer, bert_model=bert_model,
                                      mask=mask, label=label, caption=caption,
                                      mask_token=mask_token, prompt_token=prompt_token)
        print(prompt)
