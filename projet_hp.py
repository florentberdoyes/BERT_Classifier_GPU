import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import time

import pandas as pd
df = pd.read_csv("IMDB_Dataset_nettoye.csv", encoding='utf-8', quoting=1, on_bad_lines = 'skip', engine = "python")
df.head()

df['length'] = df['review'].apply(len)
df.head()

df = df[(df['length'] >= 150) & (df['length'] <= 512)]
df.count()

df.dtypes

df["review"] = df["review"].astype(str)
class_mapping = {"negative": 0, "positive": 1}
df["sentiment"] = df["sentiment"].map(class_mapping).astype(int)
df.head()

import os

def print_model_size(model, model_name):
    torch.save(model.state_dict(), f"{model_name}.pth")
    size = os.path.getsize(f"{model_name}.pth") / 1e6  # size in MB
    print(f"{model_name} size: {size:.2f} MB")
    os.remove(f"{model_name}.pth")

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(model_name)

print_model_size(model, "Original BERT")

torch.save(model.state_dict(), "model.pt")

# Apply dynamic quantization on the nn.Linear modules
quantized_model = torch.quantization.quantize_dynamic(
    model,  # the original model
    {nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8,  # specify the target dtype for weights
)
quantized_model

from torch.nn.utils import prune

model_pruned = BertForSequenceClassification.from_pretrained(model_name)

parameters_to_prune = (
    (model_pruned.bert.embeddings.word_embeddings, 'weight'),
    (model_pruned.bert.embeddings.position_embeddings, 'weight'),
    (model_pruned.bert.embeddings.token_type_embeddings, 'weight'),
)

# Apply pruning to the specified parameters
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.9,
)

for module, param in parameters_to_prune:
    prune.remove(module, param)

torch.save(model_pruned.state_dict(), "model_pruned.pt")

model_pruned = BertForSequenceClassification.from_pretrained(model_name)
model_pruned.load_state_dict(torch.load("model_pruned.pt"))

quantized_model_pruned = torch.quantization.quantize_dynamic(
    model_pruned,  # the original model
    {nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8,  # specify the target dtype for weights
)

model_prunedGPU = BertForSequenceClassification.from_pretrained(model_name)
model_prunedGPU.load_state_dict(torch.load("model_pruned.pt"))

quantized_model_pruned = torch.quantization.quantize_dynamic(
    model_prunedGPU,  # the original model
    {nn.Linear},  # specify the layers to quantize
    dtype=torch.qint8,  # specify the target dtype for weights
)

print_model_size(model, "Original BERT")
print_model_size(model_pruned, "Pruned BERT")
print_model_size(quantized_model,"Quantized BERT")
print_model_size(quantized_model_pruned, "Quantized + Pruned BERT")

import zipfile

with zipfile.ZipFile("model.zip", "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("model.pt")
print("Modèle normal:", os.path.getsize("model.zip") / 1e6, "MB")

with zipfile.ZipFile("model_pruned.zip", "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("model_pruned.pt")
print("Modèle pruné:", os.path.getsize("model_pruned.zip") / 1e6, "MB")

classes = [0,0,2,1,1]

def predict_cpu(text, model, classes):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    return classes[predicted_class]

def predict_gpu(text, model, classes):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    return classes[predicted_class]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model.to("cpu")

modelGPU = BertForSequenceClassification.from_pretrained(model_name)
modelGPU.to(device)

quantized_model.to("cpu")

model_pruned.to("cpu")

model_prunedGPU = BertForSequenceClassification.from_pretrained(model_name)
model_prunedGPU.to(device)

quantized_model_pruned.to("cpu")

RANGE = range(0,5000,500)

def tab_temps_cpu(n, df, model):
  tab = []
  for i in RANGE:
    df_copy = df.copy()
    df_copy = df_copy.head(i)
    start_time = time.time()
    df_copy["predict_model"] = df_copy["review"].apply(lambda x: predict_cpu(x, model, classes))
    end_time = time.time()
    tab.append(end_time - start_time)
  return tab, df_copy

def tab_temps_gpu(n, df, model):
  tab = []
  for i in RANGE:
    df_copy = df.copy()
    df_copy = df_copy.head(i)
    start_time = time.time()
    df_copy["predict_model"] = df_copy["review"].apply(lambda x: predict_gpu(x, model, classes))
    end_time = time.time()
    tab.append(end_time - start_time)
  return tab

n=len(df)
tab_model_cpu, df_model = tab_temps_cpu(n, df, model)
tab_quantized_model_cpu, df_quantized_model = tab_temps_cpu(n, df, quantized_model)
tab_model_pruned_cpu, df_model_pruned = tab_temps_cpu(n, df, model_pruned)
tab_quantized_model_pruned_cpu, df_quantized_model_pruned = tab_temps_cpu(n, df, quantized_model_pruned)

from matplotlib import pyplot as plt
plt.plot(RANGE, tab_model_cpu, label="Model")
plt.plot(RANGE, tab_quantized_model_cpu, label="Quantized Model")
plt.plot(RANGE, tab_model_pruned_cpu, label="Pruned Model")
plt.plot(RANGE, tab_quantized_model_pruned_cpu, label="Quantized + Pruned")
plt.legend()
plt.title("Comparaison entre les modèles (Calculs sur CPU)")
plt.xlabel("Taille du dataset")
plt.ylabel("Temps d'exécution (en secondes)")
plt.savefig("comparaison_modeles_cpu.png", dpi=300)
plt.show()

tab_model_gpu = tab_temps_gpu(n, df, modelGPU)
# tab_quantized_model_gpu = tab_temps_gpu(n, df, quantized_model)
tab_model_pruned_gpu = tab_temps_gpu(n, df, model_prunedGPU)
# tab_quantized_model_pruned_gpu = tab_temps_gpu(n, df, quantized_model_pruned)

plt.plot(RANGE, tab_model_gpu, label="Model")
# plt.plot(RANGE, tab_quantized_model_gpu, label="Quantized Model")
plt.plot(RANGE, tab_model_pruned_gpu, label="Pruned Model")
# plt.plot(RANGE, tab_quantized_model_pruned_gpu, label="Quantized + Pruned")
plt.legend()
plt.title("Comparaison entre les modèles (Calculs sur GPU)")
plt.xlabel("Taille du dataset")
plt.ylabel("Temps d'exécution (en secondes)")
plt.savefig("comparaison_modeles_gpu.png", dpi=300)
plt.show()

plt.plot(RANGE, tab_model_cpu, label="Model CPU")
plt.plot(RANGE, tab_model_gpu, label="Model GPU")
plt.legend()
plt.title("Comparaison entre les modèles (CPU vs GPU)")
plt.xlabel("Taille du dataset")
plt.ylabel("Temps d'exécution (en secondes)")
plt.savefig("comparaison_modeles_cpu_gpu.png", dpi=300)
plt.show()

plt.plot(RANGE, tab_model_pruned_cpu, label="Model CPU")
plt.plot(RANGE, tab_model_pruned_gpu, label="Model GPU")
plt.legend()
plt.title("Comparaison entre les modèles prunés (CPU vs GPU)")
plt.xlabel("Taille du dataset")
plt.ylabel("Temps d'exécution (en secondes)")
plt.savefig("comparaison_modeles_cpu_gpu_pruned.png", dpi=300)
plt.show()

df_model = df_model[df_model["predict_model"]!=2]
df_model2 = df_model[df_model["sentiment"]==df_model['predict_model']]
result = df_model2.count()/df_model.count()*100
result.values[0]

df_quantized_model = df_quantized_model[df_quantized_model["predict_model"]!=2]
df_quantized_model2 = df_quantized_model[df_quantized_model["sentiment"]==df_quantized_model['predict_model']]
result_quantized = df_quantized_model2.count()/df_quantized_model.count()*100
result_quantized.values[0]

df_pruned_model = df_model_pruned[df_model_pruned["predict_model"]!=2]
df_pruned_model2 = df_pruned_model[df_pruned_model["sentiment"]==df_pruned_model['predict_model']]
result_pruned = df_pruned_model2.count()/df_pruned_model.count()*100
result_pruned.values[0]

df_quantized_model_pruned = df_quantized_model_pruned[df_quantized_model_pruned["predict_model"]!=2]
df_quantized_model_pruned2 = df_quantized_model_pruned[df_quantized_model_pruned["sentiment"]==df_quantized_model_pruned['predict_model']]
result_quantized_model_pruned = df_quantized_model_pruned2.count()/df_quantized_model_pruned.count()*100
result_quantized_model_pruned.values[0]
