import pandas as pd
import faiss
import numpy as np
import requests


d = 4096
index=faiss.IndexFlatL2(d)
def create_textual_representations(row):
    textual_representations = f""" type:{row['type']},
    title :{row['title']},
    director:{row['director']}, 
    country:{row['country']}, 
    release_year:{row['release_year']}, 
    rating:{row['rating']},   
    description:{row['description']}
    """
    return textual_representations

def generate_embeddings(row):
    X=np.zeros((len(df['textual_representations']),d),dtype='float32')
    for i , representations in enumerate(row):
        print(f'Processed {i} instances')
        res = requests.post(url="http://localhost:11434/api/embeddings/",
                            json={
                                'model': 'llama2',
                                'prompt': representations
                            })
        embedding= res.json()['embedding']
        X[i]=np.array(embedding)
        return X


df=pd.read_csv('netflix_titles.csv')
df=df.dropna()
df['textual_representations']=df.apply(create_textual_representations, axis=1)
index.add(generate_embeddings(df['textual_representations']))
# saving the embedding locally
faiss.write_index(index,'index')
index=faiss.read_index('index')

# picking a random movie to test the code
search = df.iloc[1358]


#creating embeddings for search data
res = requests.post(url="http://localhost:11434/api/embeddings/",
                        json={
                            'model': 'llama2',
                            'prompt': search['textual_representations']
                        })
embedding=np.array([res.json()['embedding']],dtype='float32')
D,I = index.search(embedding, 4)
best_matches=np.array(df['textual_representations'])[I.flatten()]

for match in best_matches:
    print('Movie')
    print(match)

"""
Movie
 type:TV Show,
    title :Inhuman Resources,
    director:Ziad Doueiri, 
    country:France, 
    release_year:2020, 
    rating:TV-MA,   
    description:Alain Delambre, unemployed and 57, is lured by an attractive job opening. But things get ugly when he realizes he’s a pawn in a cruel corporate game.
    
Movie
 type:Movie,
    title :I Am not an Easy Man,
    director:Eleonore Pourriat, 
    country:France, 
    release_year:2018, 
    rating:TV-MA,   
    description:A shameless chauvinist gets a taste of his own medicine when he wakes up in a world dominated by women and locks horns with a powerful female author.
    
Movie
 type:TV Show,
    title :Old Money,
    director:David Schalko, 
    country:United States, 
    release_year:2015, 
    rating:TV-MA,   
    description:Backstabbing, blackmail and revenge consume the dysfunctional family of a wealthy patriarch as they compete to find him a new liver at all costs.
    
Movie
 type:Movie,
    title :The Boss's Daughter,
    director:Olivier Loustau, 
    country:France, 
    release_year:2015, 
    rating:TV-MA,   
    description:While working together, a married textile foreman and his boss’s daughter have a torrid love affair, stirring up hostility among the factory crew.





"""