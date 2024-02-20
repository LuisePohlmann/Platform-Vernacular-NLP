#Hashtags
import pandas as pd
import spacy 
from collections import Counter

#df=pd.read_csv("data_X.csv", sep=";")
df=pd.read_csv("data_Instagram.csv", sep=";")

language_model = "de_core_news_lg"
nlp = spacy.load(language_model)  

placing =[]

def find_hashtags(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    for i in range(len(sentences)):
        if "#" in str(sentences[i]):
            count=i+1
            placing.append(count)
    return placing

lst=[]

#df["placing"]=df["text"].apply(find_hashtags)
df["placing"]=df["caption"].apply(find_hashtags)

#placing= df.loc[1828, "placing"]
placing=df.loc[438, "placing"]

counts = Counter(placing)
df_counts=pd.DataFrame(counts.items())

#df_counts.to_csv("hashtags_X.csv")
df_counts.to_csv("hashtags_Insta.csv")