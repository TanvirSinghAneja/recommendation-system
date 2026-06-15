import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(stop_words='english')
from sklearn.metrics.pairwise import cosine_similarity

reviewed_book=pd.read_csv('reviewed_book.csv')

popular=reviewed_book.copy()
# Popularity 
R=popular.groupby(['Book-Title','Book-Author'])['Book-Rating'].mean()
v=popular.groupby(['Book-Title','Book-Author'])['Book-Rating'].count()
C=popular['Book-Rating'].mean()
all_cnt=len(popular)
uniq_cnt=len(popular['Book-Title'].unique())
m=all_cnt/uniq_cnt

W=(((v*R)+(C*m))/(v+m)).sort_values(ascending=False).reset_index().head(10)

book_info = reviewed_book[
    ['Book-Title','Book-Author','Image-URL-L']
].drop_duplicates(subset='Book-Title')
popular_books = W.merge(
    book_info,
    on=['Book-Title','Book-Author'],
    how='left'
)

data=reviewed_book.copy()
# Cosine Similairity Collaborative
pivot_cos=data.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pivot_cos.fillna(0,axis=1,inplace=True)
cos=cosine_similarity(pivot_cos)
pivot_cs_rec=pd.DataFrame(cos,index=pivot_cos.index.unique(),columns=pivot_cos.index.unique())
pivot_cs_rec.head()

# Cosine Similairity Content
content=data[['Book-Title','Book-Author','Publisher']].drop_duplicates(subset='Book-Title')
content['content']=content['Book-Title'].str.lower()+' '+content['Book-Author'].str.lower()+' '+content['Publisher'].str.lower()
tfidf_content=tfidf.fit_transform(content['content'])
content_df=cosine_similarity(tfidf_content)
content_df=pd.DataFrame(data=content_df,index=content['Book-Title'],columns=content['Book-Title'])

# Hybrid
common_book=pivot_cs_rec.index.intersection(content_df.index)
cos_common=pivot_cs_rec.loc[common_book,common_book]
con_common=content_df.loc[common_book,common_book]

st.title('Book Recommendation System')
book=st.selectbox('Select a book ....',options=sorted(reviewed_book['Book-Title'].unique()))
st.write(book)
same_cs=[col for col in cos_common.columns if col.lower()==book.lower()]
same_con=[col for col in con_common.columns if col.lower()==book.lower()]

st.subheader("🔥 Popular Books")

for i in range(0,10,5):
    cols = st.columns(5)

    for col, (_, row) in zip(cols, popular_books.iloc[i:i+5].iterrows()):
        with col:
           img_url = row['Image-URL-L']

            if pd.notna(img_url):
                st.image(img_url, use_container_width=True)
            st.caption(row['Book-Title'])
            st.write(row['Book-Author'])

if same_cs and same_con:
  hybrid=(0.5*cos_common[same_cs[0]])+(0.5*con_common[same_con[0]])
  st.dataframe(hybrid.sort_values(ascending=False)[1:6])
else:
  print('Book not Found')
