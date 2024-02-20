import spacy
import pandas as pd

nlp = spacy.load("de_core_news_lg")
df=pd.read_csv("data_X.csv", sep= ";")
#df=pd.read_csv("data_Instagram.csv", sep= ";")

df = df.loc[df['opinion'] != 'Advertisement']
df = df.loc[df["topic"]!= "Anzeige"]
   
def get_omission_of_article(text):
    if pd.isna(text)==True:
        return "no text"
    else:
        doc = nlp(text)
        structure = []
        for token in doc:
            structure.append(token.pos_)
            if token.text == ".":
                break
        for i in range(len(structure)):
            if structure[i]=="NOUN":
                if structure[i-1] != "DET" and structure[i-1] != "NUM" and structure[i-1]!="ADP":
                    if structure[i-1]=="ADJ":
                        if structure[i-2] != "DET" and structure[i-2] != "NUM" and structure[i-2]!="ADP":
                            return "omission of article"
                        else:
                            return "all articles included"
            elif "NOUN" not in structure:
                return "no nouns"
    

def get_missing_verb(text):
    if pd.isna(text)==True:
        return "no text"
    else:
        doc = nlp(text)
        structure = []
        for token in doc:
            structure.append(token.pos_)
            if token.text == ".":
                break
            if token.text == ":" or token.text == "!" or token.text =="-" or token.text == "\n":
                if "VERB" not in structure:
                    return "verbless clause"
                else:
                    return "all verbs included"
                
     
df["omission of article"] = df["text"].apply(get_omission_of_article)
#df["omission of article image"] = df["Image Text"].apply(get_omission_of_article)
#df["omission of article caption"] = df["caption"].apply(get_omission_of_article)
df["verbless clause"] = df["text"].apply(get_missing_verb)
#df["verbless clause image"] = df["Image Text"].apply(get_missing_verb)
#df["verbless clause caption"] = df["caption"].apply(get_missing_verb)

df.to_csv("X_headlines.csv")
df.to_csv("Insta_headlines.csv")