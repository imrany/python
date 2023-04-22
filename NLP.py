import pandas as pd
from transformers import AutoTokenizer
df=pd.read_csv("./data.csv")
print(df.head())

tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

def process_data(row):
    text=row['name']
    text=str(text)
    text=' '.join(text.split())

    encodings=tokenizer(text, padding='max_length', truncation=True, max_length=128)

    label=0
    if row['student']:
        label+=1

    encodings['label']=label
    encodings['text']=text

    return encodings

print(process_data({
    "name":"neema",
    "student":"negative"
}))