import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import time
import pandas as pd
import re


df = pd.read_csv("IMDB Dataset.csv", encoding='utf-8', quoting=1, on_bad_lines = 'skip', engine = "python")


def remove_html(raw_string):
    pattern = r'<.*?>'
    clean_text = re.sub(pattern, '', raw_string)
    return clean_text

df['review']=df['review'].apply(lambda x:remove_html(x))
df['review'] = df['review'].apply(lambda x: '.'.join(x.split('.')[:2]).strip() + '.' if '.' in x else x)
df['length'] = df['review'].apply(len)

df = df[(df['length'] >= 150) & (df['length'] <= 512)]


df["review"] = df["review"].astype(str)
class_mapping = {"negative": 0, "positive": 1}
df["sentiment"] = df["sentiment"].map(class_mapping).astype(int)
df.head()

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Apply dynamic quantization on the nn.Linear modules
quantized_model = torch.quantization.quantize_dynamic(
    model,  # the original model
    {nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8,  # specify the target dtype for weights
)
quantized_model

from torch.nn.utils import prune

parameters_to_prune = (
    (model.bert.embeddings.word_embeddings, 'weight'),
    (model.bert.embeddings.position_embeddings, 'weight'),
    (model.bert.embeddings.token_type_embeddings, 'weight'),
)

# Apply pruning to the specified parameters
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

model_pruned = BertForSequenceClassification.from_pretrained(model_name)

quantized_model_pruned = torch.quantization.quantize_dynamic(
    model_pruned,  # the original model
    {nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8,  # specify the target dtype for weights
)

import os

def print_model_size(model, model_name):
    torch.save(model.state_dict(), f"{model_name}.pth")
    size = os.path.getsize(f"{model_name}.pth") / 1e6  # size in MB
    print(f"{model_name} size: {size:.2f} MB")
    os.remove(f"{model_name}.pth")

print_model_size(model, "Original BERT")
print_model_size(model_pruned, "Pruned BERT")
print_model_size(quantized_model_pruned, "Quantized + Pruned BERT")
print_model_size(quantized_model,"Quantized BERT")




classes = [0,0,2,1,1]

def predict(text, model, classes):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    return classes[predicted_class]


model.to(device)
start_time = time.time()
df["predicted_model"] = df["review"].apply(lambda x: predict(x, model, classes))
end_time = time.time()
print(f"Temps d'exécution (modèle normal) : {end_time - start_time:.4f} secondes")

quantized_model.to(device)
start_time = time.time()
df["quantized_predicted_model"] = df["review"].apply(lambda x: predict(x, quantized_model, classes))
end_time = time.time()
print(f"Temps d'exécution (modèle quantifié) : {end_time - start_time:.4f} secondes")

model_pruned.to(device)
start_time = time.time()
df["pruned_predicted_model"] = df["review"].apply(lambda x: predict(x, model_pruned, classes))
end_time = time.time()
print(f"Temps d'exécution (modèle pruné) : {end_time - start_time:.4f} secondes")

quantized_model_pruned.to(device)
start_time = time.time()
df["quantized_pruned_predicted_model"] = df["review"].apply(lambda x: predict(x, quantized_model_pruned, classes))
end_time = time.time()
print(f"Temps d'exécution (modèle quantifié + pruné) : {end_time - start_time:.4f} secondes")

df_model = df[df["predicted_model"]!=2]
df_model2 = df_model[df_model["sentiment"]==df_model['predicted_model']]
result = df_model2.count()/df_model.count()*100
print(result.values[0])

df_quantized_model = df[df["quantized_predicted_model"]!=2]
df_quantized_model2 = df_quantized_model[df_quantized_model["sentiment"]==df_quantized_model['quantized_predicted_model']]
result_quantized = df_quantized_model2.count()/df_quantized_model.count()*100
print(result_quantized.values[0])

df_pruned_model = df[df["pruned_predicted_model"]!=2]
df_pruned_model2 = df_pruned_model[df_pruned_model["sentiment"]==df_pruned_model['pruned_predicted_model']]
result_pruned = df_pruned_model2.count()/df_pruned_model.count()*100
print(result_pruned.values[0])

df_quantized_model_pruned = df[df["quantized_pruned_predicted_model"]!=2]
df_quantized_model_pruned2 = df_quantized_model_pruned[df_quantized_model_pruned["sentiment"]==df_quantized_model_pruned['quantized_pruned_predicted_model']]
result_quantized_model_pruned = df_quantized_model_pruned2.count()/df_quantized_model_pruned.count()*100
print(result_quantized_model_pruned.values[0])