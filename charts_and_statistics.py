#!/usr/bin/env python
# coding: utf-8

# # Results
# 
# 0. [A First Look](#zero-bullet)
# 1. [Statistics](#first-bullet)
# 2. [Topics](#second-bullet)
# 3. [Hashtags](#ninth-bullet)
# 4. [Regional](#fourth-bullet)
# 5. [Videos](#tenth-bullet)
# 6. [Similarity to Headlines](#eleventh-bullet)
# 7. [Sentence Structure](#twelfth-bullet)
# 8. [Question Marks and Interaction](#seventh-bullet)
# 9. [Emojis](#eigth-bullet)
# 10. [Personal](#third-bullet)
# 11. [Opinion](#fifth-bullet)
# 12. [Emotional](#sixth-bullet)
# 

# ## 0. A First Look <a class="anchor" id="zero-bullet"></a>

# In[5]:


import pandas as pd
from scipy import stats
from scipy.stats import f_oneway
import advertools as adv
import emoji
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import numpy as np
from collections import Counter

df_X = pd.read_csv("data_X_final.csv", sep=";")
df_Insta = pd.read_csv("data_Instagram_final.csv", sep=";")
df_Insta=df_Insta.drop(390)
merged_df = pd.read_csv("merged_df_final.csv", sep=";")


# In[62]:


print(len(df_X))
print(len(df_Insta))


# In[84]:


#average amount of words
words=0
for i in df_Insta["caption"]:
    if pd.isna(i)==False:
        i = i.split(" ")
        for j in i:
            words += 1

average_word_number = words/ len(df_Insta)
print(round(average_word_number, 2))


# In[85]:


#average amount of words
words=0
no_text = 0
for i in df_Insta["Image Text"]:
    if pd.isna(i)==False:
        if i:
            i = i.split(" ")
            for j in i:
                words += 1
        else:
            no_text += 1

average_word_number = words/ (len(df_Insta)-no_text)
print(round(average_word_number, 2))


# In[65]:


#average amount of words
words=0
for i in df_X["text"]:
    if pd.isna(i)==False:
        i = i.split(" ")
        for j in i:
            words += 1

average_word_number = words/ len(df_X)
print(round(average_word_number, 2))


# In[66]:


print(round(df_X["no_of_comments"].mean(),2))
print(round(df_X["no_of_likes"].mean(),2))
print(df_X["bookmark_count"].mean())
print(df_X["retweet_count"].mean())
print(df_X["view_count"].mean())


# In[82]:


print(round(df_Insta["no_of_comments"].mean(),2))
print(round(df_Insta["no_of_likes"].mean(),2))


# In[71]:


len(df_Insta.loc[df_Insta["corresponding article"]=="yes"])/len(df_Insta)*100


# ## 1. Statistics <a class="anchor" id="first-bullet"></a>
# 
# 

# In[14]:


#Statistics X
df = df_X

t_test_columns = ["emotional", "personal", "regional", "verbless clause", "omission of article", "question", "emoji_count", "sentence_structure"]
results=[]

list_for_kruskal = []
list_for_kruskal_2 = []
list_for_kruskal_3 = []
list_for_kruskal_4 = []
list_for_kruskal_5 = []
list_for_kruskal_6 = []

for i in df["topic"].unique():
    x= df.loc[df["topic"]==i, "no_of_likes"]
    list_for_kruskal.append(x)

for i in df["topic"].unique():
    x= df.loc[df["topic"]==i, "no_of_comments"]
    list_for_kruskal_2.append(x)
    
for i in df["opinion"].unique():
    x= df.loc[df["opinion"]==i, "no_of_likes"]
    list_for_kruskal_3.append(x)

for i in df["opinion"].unique():
    x= df.loc[df["opinion"]==i, "no_of_comments"]
    list_for_kruskal_4.append(x)

for i in df["media_type"].unique():
    x= df.loc[df["media_type"]==i, "no_of_likes"]
    list_for_kruskal_5.append(x)

for i in df["media_type"].unique():
    x= df.loc[df["media_type"]==i, "no_of_comments"]
    list_for_kruskal_6.append(x)
        

k_1= stats.kruskal(list_for_kruskal[0],list_for_kruskal[1],list_for_kruskal[2],list_for_kruskal[3],list_for_kruskal[4],list_for_kruskal[5],list_for_kruskal[6],list_for_kruskal[7],list_for_kruskal[8],list_for_kruskal[9],list_for_kruskal[10],list_for_kruskal[11])
k_2= stats.kruskal(list_for_kruskal_2[0],list_for_kruskal_2[1],list_for_kruskal_2[2],list_for_kruskal_2[3],list_for_kruskal_2[4],list_for_kruskal_2[5],list_for_kruskal_2[6],list_for_kruskal_2[7],list_for_kruskal_2[8],list_for_kruskal_2[9],list_for_kruskal_2[10],list_for_kruskal_2[11])
dct = {"column": "topic", "p-value likes": k_1[1], "p-value comments": k_2[1]}
results.append(dct)


k_3= stats.kruskal(list_for_kruskal_3[0],list_for_kruskal_3[1],list_for_kruskal_3[2],list_for_kruskal_3[3])
k_4= stats.kruskal(list_for_kruskal_4[0],list_for_kruskal_4[1],list_for_kruskal_4[2],list_for_kruskal_4[3])
dct2 = {"column": "opinion", "p-value likes": k_3[1], "p-value comments": k_4[1]}
results.append(dct2)

k_5= stats.kruskal(list_for_kruskal_5[0],list_for_kruskal_5[1],list_for_kruskal_5[2])
k_6= stats.kruskal(list_for_kruskal_6[0],list_for_kruskal_6[1],list_for_kruskal_6[2])
dct3 = {"column": "media_type", "p-value likes": k_5[1], "p-value comments": k_6[1]}
results.append(dct3)


for i in t_test_columns:
    category_1 = df.loc[df[i]==df[i].unique()[0], "no_of_likes"]
    mean_1 = df.loc[df[i]==df[i].unique()[0], "no_of_likes"].mean()
    category_2 = df.loc[df[i]==df[i].unique()[1], "no_of_likes"]
    mean_2 = df.loc[df[i]==df[i].unique()[1], "no_of_likes"].mean()
    
    category_3 = df.loc[df[i]==df[i].unique()[0], "no_of_comments"]
    mean_1 = df.loc[df[i]==df[i].unique()[0], "no_of_comments"].mean()
    category_4 = df.loc[df[i]==df[i].unique()[1], "no_of_comments"]
    mean_2 = df.loc[df[i]==df[i].unique()[1], "no_of_comments"].mean()
    dct = {"column": i, "p-value likes": stats.mannwhitneyu(category_1, category_2, alternative='two-sided')[1], "p-value comments": stats.mannwhitneyu(category_3, category_4, alternative='two-sided')[1]} #df[i].unique()[0]: mean_1, df[i].unique()[1]: mean_2}
    results.append(dct)

    
data_df=pd.DataFrame(results)
data_df["p-value likes"] = data_df["p-value likes"]#round(data_df["p-value likes"],6)
data_df["p-value comments"] = data_df["p-value comments"]#round(data_df["p-value comments"],6)
data_df["type"] = "X"
                       

significant_df=data_df.loc[(data_df["p-value likes"] <= 0.05) | (data_df["p-value comments"] <= 0.05)]
significant_df
print(data_df)


# In[15]:


#Statistics Instagram
df = df_Insta

t_test_columns = ["emotional", "personal", "regional", "interaction", "question", "verbless clause caption", "is_video", "verbless clause image", "omission of article caption", "omission of article image", "emoji_count", "sentence_structure"]
results=[]

list_for_kruskal = []
list_for_kruskal_2 = []
list_for_kruskal_3 = []
list_for_kruskal_4 = []


for i in df["topic"].unique():
    x= df.loc[df["topic"]==i, "no_of_likes"]
    list_for_kruskal.append(x)

for i in df["topic"].unique():
    x= df.loc[df["topic"]==i, "no_of_comments"]
    list_for_kruskal_2.append(x)
    
for i in df["opinion"].unique():
    x= df.loc[df["opinion"]==i, "no_of_likes"]
    list_for_kruskal_3.append(x)

for i in df["opinion"].unique():
    x= df.loc[df["opinion"]==i, "no_of_comments"]
    list_for_kruskal_4.append(x)
    


k_1= stats.kruskal(list_for_kruskal[0],list_for_kruskal[1],list_for_kruskal[2],list_for_kruskal[3],list_for_kruskal[4],list_for_kruskal[5],list_for_kruskal[6],list_for_kruskal[7],list_for_kruskal[8],list_for_kruskal[9],list_for_kruskal[10],list_for_kruskal[11])
k_2= stats.kruskal(list_for_kruskal_2[0],list_for_kruskal_2[1],list_for_kruskal_2[2],list_for_kruskal_2[3],list_for_kruskal_2[4],list_for_kruskal_2[5],list_for_kruskal_2[6],list_for_kruskal_2[7],list_for_kruskal_2[8],list_for_kruskal_2[9],list_for_kruskal_2[10],list_for_kruskal_2[11])
dct = {"column": "topic", "p-value likes": k_1[1], "p-value comments": k_2[1]} #df[i].unique()[0]: mean_1, df[i].unique()[1]: mean_2}
results.append(dct)

k_3= stats.kruskal(list_for_kruskal_3[0],list_for_kruskal_3[1],list_for_kruskal_3[2],list_for_kruskal_3[3],list_for_kruskal_3[4])
k_4= stats.kruskal(list_for_kruskal_4[0],list_for_kruskal_4[1],list_for_kruskal_4[2],list_for_kruskal_4[3],list_for_kruskal_4[4])
dct2 = {"column": "opinion", "p-value likes": k_3[1], "p-value comments": k_4[1]}
results.append(dct2)



for i in t_test_columns:
    category_1 = df.loc[df[i]==df[i].unique()[0], "no_of_likes"]
    mean_1 = df.loc[df[i]==df[i].unique()[0], "no_of_likes"].mean()
    category_2 = df.loc[df[i]==df[i].unique()[1], "no_of_likes"]
    mean_2 = df.loc[df[i]==df[i].unique()[1], "no_of_likes"].mean()
    
    category_3 = df.loc[df[i]==df[i].unique()[0], "no_of_comments"]
    mean_1 = df.loc[df[i]==df[i].unique()[0], "no_of_comments"].mean()
    category_4 = df.loc[df[i]==df[i].unique()[1], "no_of_comments"]
    mean_2 = df.loc[df[i]==df[i].unique()[1], "no_of_comments"].mean()
    dct = {"column": i, "p-value likes": stats.mannwhitneyu(category_1, category_2, alternative='two-sided')[1], "p-value comments": stats.mannwhitneyu(category_3, category_4, alternative='two-sided')[1]} #df[i].unique()[0]: mean_1, df[i].unique()[1]: mean_2}
    results.append(dct)

    
data_df_1=pd.DataFrame(results)
data_df_1["p-value likes"] = data_df_1["p-value likes"]#round(data_df_1["p-value likes"],6)
data_df_1["p-value comments"] = data_df_1["p-value comments"]#round(data_df_1["p-value comments"],6)
data_df_1["type"] = "Instagram"

significant_df=data_df.loc[(data_df["p-value likes"] <= 0.05) | (data_df["p-value comments"] <= 0.05)]
#significant_df
print(data_df_1)


# In[87]:


merged_stats = pd.concat([data_df, data_df_1])
merged_stats.loc[3, "column"]="personal"
merged_stats = merged_stats.reset_index(drop=True)

sns.set(rc={'figure.figsize':(15, 8)})
sns.set_style("whitegrid", {"grid.linestyle": ":"})
colors = ["#00acee", "#C13584"]
sns.set_palette(sns.color_palette(colors))

ax = sns.scatterplot(data=merged_stats, x= "p-value likes", y="p-value comments", hue= "type", s=90)

for idx, row in merged_stats.iterrows():
    if idx == 2:
        ax.annotate(row['column'], (row['p-value likes'], row['p-value comments']+0.06))
    elif idx == 0:
        ax.annotate(row['column'], (row['p-value likes'], row['p-value comments']+0.03))
    elif idx == 12:
        ax.annotate(row['column'], (row['p-value likes'], row['p-value comments']+0.045))
    else:
        ax.annotate(row['column'], (row['p-value likes'], row['p-value comments']+0.015))

lines=[[[0.05, 0.05],[0,1], "p-value 0.05"],[[0,1],[0.05, 0.05], "p-value 0.05"]]
for i in lines:   
    line = mlines.Line2D(i[0], i[1], color='red', label=i[2], linewidth=0.3)
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)

ax.axes.set_xlim(0, 1)
ax.axes.set_ylim(0, 1)
plt.title("Correlation between the number of likes / the number of comments and the markers of platform vernacular")
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/results_scatterplot.png')
plt.show()


# In[88]:


df = df_X
df["is_video"]=df["media_type"]=="video"

df1 = df_Insta
df1=df1.rename(columns={"verbless clause caption":"verbless clause", "omission of article caption":"omission of article"})

test_columns = ["emotional", "personal", "regional", "question", "verbless clause", "omission of article", "emoji_count", "is_video"]
results={}
count=0

for i in test_columns:
    count+=1
    df.dropna(subset=[i], inplace=True)  
    state = df[i].unique()[0]
    percent1 = round(len(df.loc[df[i]==state])/len(df)*100,0)
    percent2 = round(len(df1.loc[df1[i]==state])/len(df1)*100,0)   
    results[count]={"column": i, "state": state, "percent X": percent1, "percent Instagram": percent2} #"p-value": p_value}


results_df = pd.DataFrame.from_dict(results, orient="index")
results_df


# ## 2. Topics <a class="anchor" id="second-bullet"></a>

# In[89]:


df_Insta.loc[df_Insta["topic"] == "Anzeige", "topic"] = "advertisement"


desired_order = ['politics', 'sports', 'LVZ', 'mobility', 'construction & living', 'crime, disaster & accidents', 'culture', 
                 'science & education', 'health', 'other', 'business & finance', 'celebrity', 'advertisement']


fig, (ax1, ax3) = plt.subplots(2, figsize=(15, 10), sharex=True)
fig.suptitle('Average Number of Likes and Comments by Topic on X and Insta', y=1.02)
ax2 = ax1.twinx()
ax4 = ax3.twinx()

width = 0.4

# Group by topic and calculate the mean for both columns (for X dataset)
likes_mean_X = df_X.groupby("topic")["no_of_likes"].mean().sort_values(ascending=False)
likes_mean_X = likes_mean_X.append(pd.Series([0], index=['advertisement']))
comments_mean_X = df_X.groupby("topic")["no_of_comments"].mean()
comments_mean_X = comments_mean_X.append(pd.Series([0], index=['advertisement']))


# Group by topic and calculate the mean for both columns (for X dataset)
likes_mean_X = df_X.groupby("topic")["no_of_likes"].mean().reindex(desired_order).fillna(0)
comments_mean_X = df_X.groupby("topic")["no_of_comments"].mean().reindex(desired_order).fillna(0)

# Group by topic and calculate the mean for both columns (for Insta dataset)
likes_mean_Insta = df_Insta.groupby("topic")["no_of_likes"].mean().reindex(desired_order).fillna(0)
comments_mean_Insta = df_Insta.groupby("topic")["no_of_comments"].mean().reindex(desired_order).fillna(0)

# Set positions for the bars
bar_positions1_X = range(len(desired_order))
bar_positions2_X = [pos + width for pos in bar_positions1_X]

bar_positions1_Insta = [pos for pos in bar_positions1_X]
bar_positions2_Insta = [pos + width for pos in bar_positions1_X]

# Plot the first bar chart (for X dataset)
ax1.bar(bar_positions1_X, likes_mean_X, width=width, color="#00acee", label='X: Number of Likes')
ax2.bar(bar_positions2_X, comments_mean_X, width=width, color='lightblue', label='X: Number of Comments')

# Plot the second bar chart (for Insta dataset)
ax3.bar(bar_positions1_Insta, likes_mean_Insta, width=width, color="#833AB4", alpha=0.9, label='Instagram: Number of Likes')
ax4.bar(bar_positions2_Insta, comments_mean_Insta, width=width, color="#C13584", alpha=0.7, label='Instagram: Number of Comments')

# Set x-axis ticks and labels
ax3.set_xticks([pos + width/2 for pos in bar_positions1_Insta])
ax3.set_xticklabels(desired_order, rotation=90, ha='center', va='top')

# Add space between the top of the bars and the top border of the graph
ax1.margins(y=0.1)
ax2.margins(y=0.1)
ax3.margins(y=0.1)
ax4.margins(y=0.1)

ax2.grid(False)
ax4.grid(False)

ax1.set_ylabel('Number of Likes')
ax2.set_ylabel('Number of Comments')
ax3.set_ylabel('Number of Likes')
ax4.set_ylabel('Number of Comments')

ax1.legend(loc='upper right')
ax2_legend = ax2.legend(loc='upper right')
ax2_legend.set_bbox_to_anchor((1.0, 0.9))
ax3.legend(loc='upper right')
ax4_legend = ax4.legend(loc='upper right')
ax4_legend.set_bbox_to_anchor((1.0, 0.9))

for p in ax1.patches:
    ax1.annotate(round(p.get_height(), 2), (p.get_x() + p.get_width() / 2, p.get_height() + 0.2),
                ha='center', va='center', fontsize=10, color='black')

for p in ax2.patches:
    ax2.annotate(round(p.get_height(), 2), (p.get_x() + p.get_width() / 2, p.get_height() + 0.09),
                 ha='center', va='center', fontsize=10, color='black')
    
for p in ax3.patches:
    ax3.annotate(int(round(p.get_height(), 0)), (p.get_x() + p.get_width() / 2, p.get_height() + 25),
                ha='center', va='center', fontsize=10, color='black')

for p in ax4.patches:
    ax4.annotate(int(round(p.get_height(), 0)), (p.get_x() + p.get_width() / 2, p.get_height() + 2.5),
                 ha='center', va='center', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/topic_comments_likes_Combined.png')

plt.show()


# In[90]:


other_df = merged_df.copy()
other_df = other_df.loc[other_df['opinion'] != 'Advertisement']
other_df = other_df.loc[other_df["topic"]!= "Anzeige"]
total_counts = other_df['type'].value_counts()
percentage_df = pd.DataFrame()

for topic in other_df['topic'].unique():
    percentage_df[topic] = other_df[other_df['topic'] == topic]['type'].value_counts() / total_counts

percentage_df = percentage_df.transpose()
percentage_df = percentage_df.sort_values(by='X', ascending=False)

sns.set(rc={'figure.figsize': (20, 10)})
sns.set(style="whitegrid")

ax = percentage_df.plot(kind='barh', stacked=False, color=["#00acee", '#C13584'])
plt.title('Percentage of Topics by Type', fontsize=15)
plt.ylabel('Percentage', fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.tight_layout()

for p in ax.patches:
    ax.annotate(f'{int(round(p.get_width() * 100,0))}%', ((p.get_width() + 0.004), p.get_y() + p.get_height() / 2),
                ha='center', va='center', fontsize=12, color='black')

plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/01_percentage_of_topics_by_type_2.png')
plt.show()


# ## 3. Hashtags <a class="anchor" id="ninth-bullet"></a>

# In[ ]:


hashtags =[]
count=0
for i in df_X["hashtags"]:
    if type(i)==str:
        i = i.split(",")
        for j in i:
            hashtags.append(j)
                 
hashtag_counts = Counter(hashtags)
sorted_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)

for hashtag, count in sorted_hashtags[:20]:
    print(f"{hashtag}: {count}")


# In[ ]:


hashtags =[]
count=0
for i in df_Insta["caption"]:
    if type(i)==str:
        i = i.split(" ")        
        for j in i:
            if j != "":
                if j[0] =="#":
                    hashtags.append(j)

hashtag_counts = Counter(hashtags)
sorted_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)

for hashtag, count in sorted_hashtags[:20]:
    print(f"{hashtag}: {count}")


# In[ ]:


df = df_X
sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")
colors = ["#00acee"]
sns.set_palette(sns.color_palette(colors))

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["media_type"]=="photo"]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
ax = sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)    


# In[ ]:


hashtags = 0
for i in df_X.index:
    try:
        if df_X.loc[i, "text"]:
            text = df_X.loc[i, "text"]
            text = text.split(" ")
            for i in text:
                if len(i)>0:
                    if i[0]=="#":
                        hashtags += 1
    except KeyError:
        print(i)

print(hashtags)        
hashtags/len(df_X)


# In[ ]:


hashtags = 0
for i in df_Insta.index:
    try:
        if df_Insta.loc[i, "caption"]:
            text = df_Insta.loc[i, "caption"]
            text = text.split(" ")
            for i in text:
                if len(i)>0:
                    if i[0]=="#":
                        hashtags += 1
    except KeyError:
        print(i)

print(hashtags)        
hashtags/len(df_Insta)


# In[ ]:


df=pd.read_csv("hashtags_X.csv")
df1=pd.read_csv("hashtags_Insta.csv")

sns.set(rc={'figure.figsize': (10, 5)})
sns.set(style="whitegrid")

bar_width = 0.5
plt.bar(df1["0"]+bar_width, df1["1"], label="Instagram", color='#C13584', width=bar_width)
plt.bar(df["0"], df["1"], label="X", color="#00acee", width=bar_width)

plt.title('Hashtag Placement by Sentence')
plt.xlabel('Sentence Number')
plt.ylabel('Number of Hashtags overall')
plt.legend()
plt.tight_layout()

plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/hashtags.png')
plt.show()


# ## 4. Regional <a class="anchor" id="fourth-bullet"></a>

# In[ ]:


df = df_X
#df = df_Insta
mean_1 = df.loc[df["regional"]==df["regional"].unique()[0], "no_of_likes"].mean()
mean_2 = df.loc[df["regional"]==df["regional"].unique()[1], "no_of_likes"].mean()
mean_3 = df.loc[df["regional"]==df["regional"].unique()[0], "no_of_comments"].mean()
mean_4 = df.loc[df["regional"]==df["regional"].unique()[1], "no_of_comments"].mean()
print(df["regional"].unique()[0], df["regional"].unique()[1])
print(round(mean_1,2), round(mean_2,2))
print(round(mean_3,2), round(mean_4,2))


# In[ ]:


df = df_Insta
mean_1 = df.loc[df["regional"]==df["regional"].unique()[0], "no_of_likes"].mean()
mean_2 = df.loc[df["regional"]==df["regional"].unique()[1], "no_of_likes"].mean()
mean_3 = df.loc[df["regional"]==df["regional"].unique()[0], "no_of_comments"].mean()
mean_4 = df.loc[df["regional"]==df["regional"].unique()[1], "no_of_comments"].mean()
print(df["regional"].unique()[0], df["regional"].unique()[1])
print(round(mean_1,2), round(mean_2,2))
print(round(mean_3,2), round(mean_4,2))


# In[ ]:


regional_df = merged_df.copy()
total_counts = regional_df['type'].value_counts()
percentage_df = pd.DataFrame()

for opinion in regional_df['regional'].unique():
    percentage_df[opinion] = regional_df[regional_df['regional'] == opinion]['type'].value_counts() / total_counts

percentage_df = percentage_df.transpose()
percentage_df = percentage_df.sort_values(by='Instagram', ascending=False)

sns.set(rc={'figure.figsize': (10, 6)})
sns.set(style="whitegrid")

ax = percentage_df.plot(kind='bar', stacked=False, color=["#00acee", '#C13584'])
plt.title('Percentage of Regional News by Type')
plt.xlabel('Type')
plt.ylabel('Percentage')

# Add annotations to each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                ha='center', va='center', fontsize=10, color='black')

#plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/04_percentage_of_regional_news_by_type.png')
plt.show()


# In[ ]:


df = df_X

sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["regional"]=="regional"]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
ax = sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)

for p in ax.patches:
    ax.annotate(round(p.get_height(),2), (p.get_x() + p.get_width() / 2, p.get_height()+1),
                ha='center', va='center', fontsize=10, color='black')
    


# In[ ]:


df = df_Insta 

sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["regional"]=="regional"]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
ax = sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)

for p in ax.patches:
    ax.annotate(round(p.get_height(),2), (p.get_x() + p.get_width() / 2, p.get_height()+1),
                ha='center', va='center', fontsize=10, color='black')
    


# In[ ]:


df = df_X

mean_emotional = df.loc[df["regional"]=="regional", "no_of_comments"].mean()
mean_not_emotional = df.loc[df["regional"]== "(inter)national", "no_of_comments"].mean()

df=df.loc[df["no_of_comments"]<20]

sns.set(rc={'figure.figsize':(10, 6)})
sns.set(style="whitegrid")

colors = ["#0c71b5", "#65bbf5"]
sns.set_palette(sns.color_palette(colors))

sns.boxplot(data= df, x="regional", y="no_of_comments", saturation=0.5) 

plt.title('Number of Comments for regional and (inter)national posts on X')
plt.xlabel('Topic')
plt.ylabel('Number of Comments')
 
plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/regional_comments_X.png')
plt.show()

print(round(mean_emotional,2), round(mean_not_emotional,2))


# ## 5. Videos <a class="anchor" id="tenth-bullet"></a>

# In[ ]:


df =df_Insta

print(df.groupby("is_video")["caption"].count())
print(df.groupby("is_video")[["no_of_likes", "no_of_comments"]].mean())


# In[ ]:


df = df_X

df['media_type'] = df['media_type'].fillna("text")

print(df.groupby("media_type")["text"].count())
print(df.groupby("media_type")[["no_of_likes", "no_of_comments", "bookmark_count", "retweet_count","view_count"]].mean())


# In[ ]:


df = df_X
sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")
colors = ["#00acee"]
sns.set_palette(sns.color_palette(colors))

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["media_type"]=="video"]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
ax = sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)    


# In[ ]:


df = df_X
sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")
colors = ["#00acee"]
sns.set_palette(sns.color_palette(colors))

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["media_type"]=="photo"]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
ax = sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)    


# In[ ]:


df =df_Insta 
sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")
colors = ["#00acee"]
sns.set_palette(sns.color_palette(colors))

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["is_video"]==True]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
ax = sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)

for p in ax.patches:
    ax.annotate(round(p.get_height(),2), (p.get_x() + p.get_width() / 2, p.get_height()+1),
                ha='center', va='center', fontsize=10, color='black')


# In[ ]:


#df = df_X
df = df_Insta
sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")
colors = ["#00acee"]
sns.set_palette(sns.color_palette(colors))

df.dropna(subset=["regional"], inplace=True)

dct={}
count=0
for i in df["regional"].unique():
    count+=1
    x=df.loc[df["regional"]==i]
    p=x.loc[x["is_video"]==True]
    print(len(x))
    print(len(p))
    dct[count]={"regional":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["regional","average"])
ax = sns.barplot(data=df1, x="regional", y="average", order=df1.sort_values('average', ascending = False).regional)

for p in ax.patches:
    ax.annotate(round(p.get_height(),2), (p.get_x() + p.get_width() / 2, p.get_height()+1),
                ha='center', va='center', fontsize=10, color='black')
    


# In[ ]:


#df = df_X
sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")
colors = ["#00acee"]
sns.set_palette(sns.color_palette(colors))

df.dropna(subset=["regional"], inplace=True)

dct={}
count=0
for i in df["regional"].unique():
    count+=1
    x=df.loc[df["regional"]==i]
    p=x.loc[x["media_type"]=="video"]
    print(len(x))
    print(len(p))
    dct[count]={"regional":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["regional","average"])
ax = sns.barplot(data=df1, x="regional", y="average", order=df1.sort_values('average', ascending = False).regional)

for p in ax.patches:
    ax.annotate(round(p.get_height(),2), (p.get_x() + p.get_width() / 2, p.get_height()+1),
                ha='center', va='center', fontsize=10, color='black')
    


# ## 6. Similarity to Headlines <a class="anchor" id="eleventh-bullet"></a>

# In[ ]:


df=df_Insta
print(len(df.loc[df["omission of article caption"] == "omission of article"]))
print(len(df.loc[df["omission of article caption"] == "all articles included"]))
print(len(df.loc[df["omission of article image"] == "omission of article"]))
print(len(df.loc[df["omission of article image"] == "all articles included"]))


# In[ ]:


df=df_Insta
print(len(df.loc[df["verbless clause caption"] == "verbless clause"]))
print(len(df.loc[df["verbless clause caption"] == "all verbs included"]))
print(len(df.loc[df["verbless clause image"] == "verbless clause"]))
print(len(df.loc[df["verbless clause image"] == "all verbs included"]))


# In[ ]:


df2=df_X
print(len(df2.loc[df2["omission of article"] == "omission of article"]))
print(len(df2.loc[df2["omission of article"] == "all articles included"]))


# In[ ]:


df2=df_X
print(len(df2.loc[df2["verbless clause"] == "verbless clause"]))
print(len(df2.loc[df2["verbless clause"] == "all verbs included"]))


# In[ ]:


print(len(df.loc[df["verbless clause caption"] == "verbless clause"])/len(df))
print(len(df.loc[df["verbless clause caption"] == "all verbs included"])/len(df))
print(len(df.loc[df["verbless clause image"] == "verbless clause"])/len(df))
print(len(df.loc[df["verbless clause image"] == "all verbs included"])/len(df))
print(len(df2.loc[df2["verbless clause"] == "verbless clause"])/len(df2))
print(len(df2.loc[df2["verbless clause"] == "all verbs included"])/len(df2))


# In[ ]:


df1_values = {
    'verbless clause': len(df.loc[df["verbless clause caption"] == "verbless clause"]) / len(df),
    'all verbs included': len(df.loc[df["verbless clause caption"] == "all verbs included"]) / len(df)
}
df_values ={
    'verbless clause': len(df.loc[df["verbless clause image"] == "verbless clause"]) / len(df),
    'all verbs included': len(df.loc[df["verbless clause image"] == "all verbs included"]) / len(df),
}

df2_values = {
    'verbless clause': len(df2.loc[df2["verbless clause"] == "verbless clause"]) / len(df2),
    'all verbs included': len(df2.loc[df2["verbless clause"] == "all verbs included"]) / len(df2),
}

# Create dataframes
df_values_df = pd.DataFrame(df_values.values(), index=df_values.keys(), columns=['Instagram Image Text'])
df1_values_df = pd.DataFrame(df1_values.values(), index=df_values.keys(), columns=['Instagram Caption Text'])
df2_values_df = pd.DataFrame(df2_values.values(), index=df2_values.keys(), columns=['X Text'])

# Concatenate dataframes
plot_df = pd.concat([df_values_df, df1_values_df, df2_values_df], axis=1)

# Plot the bar chart with different colors for each column
sns.set(rc={'figure.figsize':(10, 6)})
sns.set(style="white")
ax = plot_df.plot(kind='bar', color=['#C13584', "#833AB4", "#00acee"])

# Customize the plot
plt.title('Percentage of Verbless Clauses')
plt.xlabel('Types of Clauses')
plt.ylabel('Percentage')
plt.xticks(rotation=0)

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                ha='center', va='center', fontsize=10, color='black')

plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/07_percentage_of_verbless_claues.png')
plt.show()


# In[ ]:


df1_values = {
    'omission of articles': len(df.loc[df["omission of article caption"] == "omission of article"]) / len(df),
    'all articles included': len(df.loc[df["omission of article caption"] == "all articles included"]) / len(df),
    'no nouns': len(df.loc[df["omission of article caption"] == "no nouns"]) / len(df),
    'no text': len(df.loc[df["omission of article caption"] == "no text"]) / len(df)
}
df_values = {
    'omission of articles': len(df.loc[df["omission of article image"] == "omission of article"]) / len(df),
    'all articles included': len(df.loc[df["omission of article image"] == "all articles included"]) / len(df),
    'no nouns': len(df.loc[df["omission of article image"] == "no nouns"]) / len(df),
    'no text': len(df.loc[df["omission of article image"] == "no text"]) / len(df)
}

df2_values = {
    'omission of articles': len(df2.loc[df2["omission of article"] == "omission of article"]) / len(df2),
    'all articles included': len(df2.loc[df2["omission of article"] == "all articles included"]) / len(df2),
    'no nouns': len(df2.loc[df2["omission of article"] == "no nouns"]) / len(df2),
    'no text': len(df2.loc[df2["omission of article"] == "no text"]) / len(df2)
}


# Create dataframes
df_values_df = pd.DataFrame(df_values.values(), index=df_values.keys(), columns=['Instagram Image Text'])
df1_values_df = pd.DataFrame(df1_values.values(), index=df1_values.keys(), columns=['Instagram Caption Text'])
df2_values_df = pd.DataFrame(df2_values.values(), index=df2_values.keys(), columns=['X Text'])

# Concatenate dataframes
plot_df = pd.concat([df_values_df, df1_values_df, df2_values_df], axis=1)

# Plot the bar chart with different colors for each column
sns.set(rc={'figure.figsize': (10, 6)})
sns.set(style="white")
ax = plt.gca()

# Define bar width
bar_width = 0.3
color=['#C13584', "#833AB4", "#00acee"]

# Plotting
for i, column in enumerate(plot_df.columns):
    x = range(len(plot_df))
    ax.bar([pos + i * bar_width for pos in x], plot_df[column], width=bar_width, label=column, color=color[i])

# Customize the plot
plt.title('Percentage of Omission of Articles')
plt.xlabel('Types of Clauses')
plt.ylabel('Percentage')
plt.xticks([pos + bar_width for pos in x], plot_df.index)
plt.legend()
plt.tight_layout()

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                ha='center', va='center', fontsize=10, color='black')

plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/07_percentage_of_omission_of_articles.png')
plt.show()


# In[ ]:


#average word length in characters
word_lengths = 0
words=0
for i in df2["text"]:
    if pd.isna(i)==False:
        i = i.split(" ")
        for j in i:
            word_lengths += len(j)
            words += 1

average_word_length_text = word_lengths/ words
print(round(average_word_length_text, 2))


# In[ ]:


#average word length in characters
word_lengths = 0
words=0
for i in df["caption"]:
    if pd.isna(i)==False:
        i = i.split(" ")
        for j in i:
            word_lengths += len(j)
            words += 1

average_word_length_text = word_lengths/ words
print(round(average_word_length_text, 2))


# In[ ]:


#average word length in characters
word_lengths = 0
words=0
for i in df["Image Text"]:
    if pd.isna(i)==False:
        i = i.split(" ")
        for j in i:
            word_lengths += len(j)
            words += 1

average_word_length_text = word_lengths/ words
print(round(average_word_length_text, 2))


# In[ ]:


df =df_Insta 
sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")
colors = ["#00acee"]
sns.set_palette(sns.color_palette(colors))

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["verbless clause caption"]=="verbless clause"]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
ax = sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)

for p in ax.patches:
    ax.annotate(round(p.get_height(),2), (p.get_x() + p.get_width() / 2, p.get_height()+1),
                ha='center', va='center', fontsize=10, color='black')


# In[ ]:


df =df_Insta 
sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")
colors = ["#00acee"]
sns.set_palette(sns.color_palette(colors))

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["omission of article caption"]=="omission of article"]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
ax = sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)

for p in ax.patches:
    ax.annotate(round(p.get_height(),2), (p.get_x() + p.get_width() / 2, p.get_height()+1),
                ha='center', va='center', fontsize=10, color='black')


# In[ ]:


df2=df_X
x=df2.loc[df2["verbless clause"] == "verbless clause"]
print(x["no_of_comments"].mean())
y=df2.loc[df2["verbless clause"] == "all verbs included"]
print(y["no_of_comments"].mean())


# In[ ]:


df2=df_X
print(len(df2.loc[df2["verbless clause"] == "verbless clause"]))
print(len(df2.loc[df2["verbless clause"] == "all verbs included"]))


# ## 7. Sentence Structure <a class="anchor" id="twelfth-bullet"></a>

# In[ ]:


df_X.dropna(subset=["ratio_main_sub"], inplace=True) 
df_Insta.dropna(subset=["ratio_main_sub"], inplace=True)

print("ratio_main_sub:", "Insta:", df_Insta["ratio_main_sub"].mean(), "X:", df_X["ratio_main_sub"].mean())
print("ratio_main:", "Insta:", df_Insta["ratio_main"].mean(), "X: ", df_X["ratio_main"].mean())
print("ratio_sub:", "Insta:", df_Insta["ratio_sub"].mean(), "X: ", df_X["ratio_sub"].mean())
print("mainclauses:", "Insta:", df_Insta["main_clauses"].mean(), "X:", df_X["main_clauses"].mean())
print("subclauses:", "Insta:", df_Insta["sub_clauses"].mean(), "X:" , df_X["sub_clauses"].mean())
print("number of sentences:", "Insta:", df_Insta["no_of_sentences"].mean(), "X:" , df_X["no_of_sentences"].mean())


# In[ ]:


print(round(len(df_X.loc[df_X["sentence_structure"]=="complex"])/len(df_X)*100, 2))
print(round(len(df_X.loc[df_X["sentence_structure"]=="simple"])/len(df_X)*100, 2))


# In[ ]:


print(round(len(df_Insta.loc[df_Insta["sentence_structure"]=="complex"])/len(df_Insta)*100, 2))
print(round(len(df_Insta.loc[df_Insta["sentence_structure"]=="simple"])/len(df_Insta)*100, 2))


# ## 8. Question Marks and Interaction <a class="anchor" id="seventh-bullet"></a>

# In[ ]:


df = df_Insta
df["question"]=df["question"].replace(0, False)
df["question"]=df["question"].replace([1,2,3,4], True)

mean_question = df.loc[df["question"]==True, "no_of_comments"].mean()
mean_no_question = df.loc[df["question"]== False, "no_of_comments"].mean()
print(mean_question, mean_no_question) 


# In[ ]:


df = df_Insta
df["question"]=df["question"].replace(0, False)
df["question"]=df["question"].replace([1,2,3,4], True)
df["interaction"]=df["interaction"].replace("yes", "interaction")
df["interaction"]=df["interaction"].replace("no", "no interaction")

df_questions = df.loc[df["question"]==True]
len(df_questions.loc[df["interaction"]=="interaction"])/ len(df_questions) *100


# In[ ]:


df=df_Insta
df["interaction"]=df["interaction"].replace("yes", "interaction")
df["interaction"]=df["interaction"].replace("no", "no interaction")
len(df.loc[df["interaction"]=="interaction"])/ len(df) *100


# In[ ]:


df = df_X
df["question"]=df["question"].replace(0, False)
df["question"]=df["question"].replace([1,2,3,4], True)

mean_question = df.loc[df["question"]==True, "no_of_comments"].mean()
mean_no_question = df.loc[df["question"]== False, "no_of_comments"].mean()
mean_question1 = df.loc[df["question"]==True, "no_of_likes"].mean()
mean_no_question1 = df.loc[df["question"]== False, "no_of_likes"].mean()
print("likes:", "with questions:", mean_question1, "without questions:", mean_no_question1) 
print("comments:", "with questions:", mean_question, "without questions:", mean_no_question)  


# In[ ]:


df = df_X

sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["question"]==True]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
ax = sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)   
plt.xticks(rotation=90)


# In[ ]:


print(round(len(df_X.loc[df_X["question"]==True])/len(df_X)*100))
print(round(len(df_Insta.loc[df_Insta["question"]==True])/len(df_Insta)*100))


# ## 9. Emojis <a class="anchor" id="eigth-bullet"></a>

# In[ ]:


print(round((len(df_Insta.loc[df_Insta["emoji_count"]==1])/len(df_Insta))*100))
print(round((len(df_X.loc[df_X["emoji_count"]==1])/len(df_X))*100,2))


# In[ ]:


df = df_Insta

mean_emotional = df.loc[df["emoji_count"]== 1, "no_of_comments"].mean()
mean_not_emotional = df.loc[df["emoji_count"]== 0, "no_of_comments"].mean()


df=df.loc[df["no_of_comments"]<100]

colors = ["#C13584", "#833AB4"]
sns.set_palette(sns.color_palette(colors))

sns.set(rc={'figure.figsize':(10, 6)})
sns.set(style="whitegrid")
sns.set_palette(sns.color_palette(colors))

sns.boxplot(data= df, x="emoji_count", y="no_of_comments", saturation=0.5) 

plt.title('Number of Comments for posts with emojis and without emojis on Insta')
plt.xlabel('Emojis')
plt.ylabel('Number of Comments')
 
plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/personal_comments_Insta.png')
plt.show()

print(round(mean_emotional,2), round(mean_not_emotional,2))


# In[ ]:


df=df_Insta        
print(len(df.loc[df["emoji_count"]==1]))


# In[ ]:


df = df_Insta 

sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["emoji_count"]==1]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
ax = sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)   
plt.xticks(rotation=90)


# In[ ]:


#Emojis
df=df_Insta

orig_list=[]

def find_emojis(text):
    orig_list.append(text)

df["caption"].apply(find_emojis)
emoji_dict = adv.extract_emoji(orig_list) 

print(emoji_dict["emoji_freq"])
print(emoji_dict["top_emoji"][:10])
print(emoji_dict['top_emoji_text'][:10])
print(emoji_dict['top_emoji_groups'][:5])#, emoji_dict['top_emoji_sub_groups'])
print(emoji_dict['overview'])


# In[ ]:


#Emojis
df=df_X
orig_list=[]

def find_emojis(text):
    orig_list.append(text)

df["text"].apply(find_emojis)

emoji_dict = adv.extract_emoji(orig_list)    
#dict_keys(['emoji', 'emoji_text', 'emoji_flat', 'emoji_flat_text', 'emoji_counts', 'emoji_freq', 'top_emoji', 'top_emoji_text', 'top_emoji_groups', 'top_emoji_sub_groups', 'overview'])
print(emoji_dict["emoji_freq"])
print(emoji_dict["top_emoji"][:10])
print(emoji_dict['top_emoji_text'])
print(emoji_dict['top_emoji_groups'][:5])#, emoji_dict['top_emoji_sub_groups'])
print(emoji_dict['overview'])


# In[ ]:


import matplotlib.pyplot as plt

df["emoji"] = emoji_dict["emoji"]
df["emoji"][8]

for i in range(len(df)):
    if not df["emoji"][i]:
        df.loc[i, "emoji_count"]=0
    else:
        df.loc[i, "emoji_count"]=1
        


df_topics= df.groupby("topic")["emoji_count"].sum()/ lengths *100
df_topics= df_topics.sort_values(ascending = False)
df_topics


# ## 10. Personal <a class="anchor" id="third-bullet"></a>

# In[91]:


df = df_X
mean_1 = df.loc[df["personal"]==df["personal"].unique()[0], "no_of_likes"].mean()
mean_2 = df.loc[df["personal"]==df["personal"].unique()[1], "no_of_likes"].mean()
mean_3 = df.loc[df["personal"]==df["personal"].unique()[0], "no_of_comments"].mean()
mean_4 = df.loc[df["personal"]==df["personal"].unique()[1], "no_of_comments"].mean()
print(df["personal"].unique()[0], df["personal"].unique()[1])
print(round(mean_1,2), round(mean_2,2))
print(round(mean_3,2), round(mean_4,2))


# In[92]:


df = df_Insta
mean_1 = df.loc[df["personal"]==df["personal"].unique()[0], "no_of_likes"].mean()
mean_2 = df.loc[df["personal"]==df["personal"].unique()[1], "no_of_likes"].mean()
mean_3 = df.loc[df["personal"]==df["personal"].unique()[0], "no_of_comments"].mean()
mean_4 = df.loc[df["personal"]==df["personal"].unique()[1], "no_of_comments"].mean()
print(df["personal"].unique()[0], df["personal"].unique()[1])
print(round(mean_1,2), round(mean_2,2))
print(round(mean_3,2), round(mean_4,2))


# In[93]:


df = df_X
percent = round(len(df.loc[df["personal"]=="personalisation"])/len(df)*100,0)
percent


# In[94]:


df = df_Insta
percent = round(len(df.loc[df["personal"]=="personalisation"])/len(df)*100,0)
percent


# In[95]:


personal_df=merged_df.copy()

total_counts = personal_df['type'].value_counts()
percentage_df = pd.DataFrame()

for i in personal_df['personal'].unique():
    percentage_df[i] = personal_df[personal_df['personal'] == i]['type'].value_counts() / total_counts
percentage_df = percentage_df.transpose()
percentage_df = percentage_df.sort_values(by='Instagram', ascending=False)

sns.set(rc={'figure.figsize':(10, 6)})
sns.set(style="whitegrid")
ax = percentage_df.plot(kind='bar', stacked=False, color=["#00acee", '#C13584'])
plt.title('Percentage of Personal Reporting by Type')
plt.xlabel('Type')
plt.ylabel('Percentage')

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                ha='center', va='center', fontsize=10, color='black')
    

plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/personal.png')
plt.show()


# In[96]:


df = df_X
sns.set(rc={'figure.figsize':(10, 6)})
sns.set(style="whitegrid")

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["personal"]=="personalisation"]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)


# In[100]:


df = df_Insta 
sns.set(rc={'figure.figsize':(10, 6)})
sns.set(style="whitegrid")

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["personal"]=="personalisation"]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)


# In[74]:


df = df_X
df.dropna(subset=["regional"], inplace=True)

sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")
colors = ["#00acee"]
sns.set_palette(sns.color_palette(colors))

dct={}
count=0
for i in df["regional"].unique():
    count+=1
    x=df.loc[df["regional"]==i]
    p=x.loc[x["personal"]=="personalisation"]
    dct[count]={"regional":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["regional","average"])
print(df1)
#sns.barplot(data=df1, x="regional", y="average", order=df1.sort_values('average', ascending = False).regional)


# In[120]:


df = df_Insta
df.dropna(subset=["regional"], inplace=True)

sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")
colors = ["#00acee"]
sns.set_palette(sns.color_palette(colors))

dct={}
count=0
for i in df["regional"].unique():
    count+=1
    x=df.loc[df["regional"]==i]
    p=x.loc[x["personal1"]=="personalisation"]
    dct[count]={"regional":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["regional","average"])
print(df1)
#sns.barplot(data=df1, x="regional", y="average", order=df1.sort_values('average', ascending = False).regional)


# ## 11. Opinion <a class="anchor" id="fifth-bullet"></a>

# In[106]:


df = df_X

unique_opinions = df["opinion"].unique()
likes_means= []
comments_means = []

for opinion in df["opinion"].unique():
    likes_mean = df.loc[df["opinion"] == opinion, "no_of_likes"].mean()
    comments_mean = df.loc[df["opinion"] == opinion, "no_of_comments"].mean()
    likes_means.append(likes_mean)
    comments_means.append(comments_mean)
    
for opinion, likes_mean, comments_mean in zip(unique_opinions, likes_means, comments_means):
    print(opinion, "Likes Mean:", round(likes_mean, 2), "Comments Mean:", round(comments_mean, 2),  "\t")


# In[108]:


x=df_X.loc[df_X["opinion"]=="Report/ Analysis"]
print(x["text"], x["no_of_likes"], x["no_of_comments"])
print(len(x))


# In[109]:


df = df_Insta

unique_opinions = df["opinion"].unique()
likes_means= []
comments_means = []

for opinion in df["opinion"].unique():
    likes_mean = df.loc[df["opinion"] == opinion, "no_of_likes"].mean()
    comments_mean = df.loc[df["opinion"] == opinion, "no_of_comments"].mean()
    likes_means.append(likes_mean)
    comments_means.append(comments_mean)
    
for opinion, likes_mean, comments_mean in zip(unique_opinions, likes_means, comments_means):
    print(opinion, "Likes Mean:", round(likes_mean, 2), "Comments Mean:", round(comments_mean, 2),  "\t")


# In[110]:


df = df_Insta

sns.set(rc={'figure.figsize':(10, 6)})
sns.set(style="whitegrid")


ax=df.groupby("opinion")["no_of_comments"].mean().sort_values(ascending = False).plot(kind="bar",legend=True, color=["#C13584"])

plt.title('Average Number of Comments for Types News on Instagram')
plt.xlabel('Types of News')
plt.ylabel('Number of Comments')
plt.tight_layout()

for p in ax.patches:
    ax.annotate(round(p.get_height(),2), (p.get_x() + p.get_width() / 2, p.get_height()+1),
                ha='center', va='center', fontsize=10, color='black')
    

plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/opinion_comments_insta.png')
plt.show()


# In[111]:


df = df_Insta 

sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")
colors = ["#00acee"]
sns.set_palette(sns.color_palette(colors))

dct={}
count=0
for i in df["topic"].unique():
    count+=1
    x=df.loc[df["topic"]==i]
    p=x.loc[x["opinion"]=="News"]
    dct[count]={"topic":i, "average": round(len(p)/len(x)*100,2)}

df1=pd.DataFrame.from_dict(dct, orient='index', columns=["topic","average"])
ax = sns.barplot(data=df1, x="topic", y="average", order=df1.sort_values('average', ascending = False).topic)   
plt.xticks(rotation=90)


# In[112]:


opinion_df=merged_df.copy()

total_counts = opinion_df['type'].value_counts()
percentage_df = pd.DataFrame()

for opinion in opinion_df['opinion'].unique():
    percentage_df[opinion] = opinion_df[opinion_df['opinion'] == opinion]['type'].value_counts() / total_counts

percentage_df = percentage_df.transpose()
percentage_df = percentage_df.sort_values(by='Instagram', ascending=False)

sns.set(rc={'figure.figsize':(15, 8)})
sns.set(style="whitegrid")
ax = percentage_df.plot(kind='bar', stacked=False, color=["#00acee", '#C13584'])
plt.title('Percentage of Opinion Pieces by Type')
plt.xlabel('Type')
plt.ylabel('Percentage')

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100:.2f}%', ((p.get_x()+p.get_width()/2), p.get_y() + p.get_height() +  0.015),
                ha='center', va='center', fontsize=12, color='black')

#plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/02_percentage_of_opinion_pieces_by_type.png') 
plt.show()


# ## 12. Emotional <a class="anchor" id="sixth-bullet"></a>

# In[ ]:


emotional_df=merged_df.copy()
emotional_df['emotional'] = emotional_df['emotional'].replace(["no ", "o"], "no")
emotional_df = emotional_df.dropna(subset=['emotional'])

total_counts = emotional_df['type'].value_counts()
percentage_df = pd.DataFrame()

for opinion in emotional_df['emotional'].unique():
    percentage_df[opinion] = emotional_df[emotional_df['emotional'] == opinion]['type'].value_counts() / total_counts
percentage_df = percentage_df.transpose()
percentage_df = percentage_df.sort_values(by='Instagram', ascending=False)

sns.set(rc={'figure.figsize':(10, 6)})
sns.set(style="whitegrid")
ax = percentage_df.plot(kind='bar', stacked=False, color=["#00acee", '#C13584'])
plt.title('Percentage of Emotional Reporting by Type')
plt.xlabel('Type')
plt.ylabel('Percentage')
plt.tight_layout()

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                ha='center', va='center', fontsize=10, color='black')

plt.savefig('C:/Users/Lui/Desktop/Leuphana/01_Bachelor Arbeit/Data/Images/06_percentage_of_emotional_reporting_by_type.png')
plt.show()


# In[ ]:


fig, (ax, ax1) = plt.subplots(1, 2)
sns.set(rc={'figure.figsize':(15, 8)})
sns.countplot(df_Insta["emotion"], label="Instagram", color='#C13584', ax=ax)
ax.set_title('Instagram')
ax.set_xlabel('Emotion')
ax.set_ylabel('Count')

sns.countplot(df_X["emotion"], label="X", color="#00acee", ax=ax1)
ax1.set_title('X')
ax1.set_xlabel('Emotion')
ax1.set_ylabel('Count')

for p in ax.patches:
    ax.annotate(p.get_height(), ((p.get_width()/2 +p.get_x()), p.get_y() + p.get_height() +0.6),
                ha='center', va='center', color='black')
for p in ax1.patches:
    ax1.annotate(p.get_height(), ((p.get_width()/2 +p.get_x()), p.get_y() + p.get_height() +1),
                ha='center', va='center', color='black')


plt.tight_layout()
plt.show()


# In[ ]:


len(df_Insta.loc[df_Insta["emotion"] == "negative"])/len(df_Insta.loc[df_Insta["emotional"]=="emotionalisation"])*100


# In[ ]:


len(df_X.loc[df_X["emotion"] == "negative"])/len(df_X.loc[df_X["emotional"]=="emotionalisation"])*100

