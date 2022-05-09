import streamlit as st
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim.corpora import Dictionary
from gensim.models import Phrases
from scipy import sparse as sp
from gensim.models import LdaModel
import pyLDAvis
from streamlit import components
import pyLDAvis.gensim_models as gensimvis
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
from gensim.models import CoherenceModel
import os
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

stop = stopwords.words('english')
new_stops = ['said','us','the','could','also']
stop.extend(new_stops)


st.header('Topic Visualizations')

def read_and_preprocess(path_of_data):
    # Reading Data and preprocessing
    data = pd.read_excel(path_of_data)
    data = data.dropna()
    data = data.reset_index(drop=True)
    return data


def display_data(data,text):
    st.subheader(text)
    st.write(data.astype(str))  # Same as st.write(df)

def clean_data(data):
    data = data[['DocumentID','Headline','BodyText']]
    data['Headline'] = data['Headline'].astype(str)
    data['BodyText'] = data['BodyText'].astype(str)
    data["text"] = data["Headline"] +" "+ data["BodyText"]
    data = data.drop(['Headline', 'BodyText'], axis=1)

    # Removing stopwords
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    # Removing punctuation
    data["text"] = data['text'].str.replace('[^\w\s]','')

    # Converting text to lowercase
    data["text"] = data["text"].str.lower()

    car_names = ['vw','kia','honda','toyota','chrysler','nissan','volkswagen','chevrolet','ford','hyundai',
    'vehicles','dodge','toyotas','daimler','audi','auto','fiat','lincoln','minivans','mazda','bmw','gm']

    for i in car_names:
        data.text = data.text.str.replace(i,'vehicle',regex=True)

    return data

def print_common_words(data):
    data['temp_list'] = data['text'].apply(lambda x:str(x).split())
    top = Counter([item for sublist in data['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(30))
    data = data.drop(['temp_list'],axis=1)
    temp.columns = ['Common_words','count']
    st.write(temp.style.background_gradient(cmap='Blues'))


### LDA part
def docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isdigit()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 3] for doc in docs]

    # Lemmatize all words in documents.
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    return docs

def explore_topic(lda_model, topic_number, topn, output=True):
    """
    accept a ldamodel, atopic number and topn vocabs of interest
    prints a formatted list of the topn terms
    """
    terms = []
    for term, frequency in lda_model.show_topic(topic_number, topn=topn):
        terms += [term]
        if output:
            st.write(u'{:20} {:.3f}'.format(term, round(frequency, 3)))

    return terms



# path_of_data = '/Users/gautamchauhan/Documents/RA_prof_vivek/Task_4_model_on_news_about_safety_topics/lda_streamlit_app/20211209-SVM_news_text.xlsx'


path = os.getcwd()
path_of_data = '20211209-SVM_news_text.xlsx'
data = read_and_preprocess(path_of_data)

display_data(data,"Printing raw data")

data = clean_data(data)

display_data(data,"Printing clean data: Removed Stopwords, Punctuation and replaced car names with 'vehicle'")

st.subheader('Printing Common words and their frequency')
print_common_words(data)


st.header('LDA')


docs = data.text

docs = docs_preprocessor(docs)

# Add bigrams and trigrams to docs (only ones that appear 10 times or more).
bigram = Phrases(docs, min_count=10)
trigram = Phrases(bigram[docs])

for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)
    for token in trigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)
st.caption('Number of unique words in initital documents: ' +str(len(dictionary)))

# Filter out words that occur less than 10 documents, or more than 20% of the documents.
dictionary.filter_extremes(no_below=10, no_above=0.2)
st.caption('Number of unique words after removing common words using Gensim Library: ' +str(len(dictionary)))

corpus = [dictionary.doc2bow(doc) for doc in docs]


# num_topics = int(st.text_input('Please input the number of topics:'))

num_topics = st.number_input('Input Number of Topics',0,20)

if num_topics:
    st.spinner('Running Model')
    num_topics = num_topics
    chunksize = 500 # size of the doc looked at every pass
    passes = 20 # number of passes through documents
    iterations = 400
    eval_every = 1  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                           alpha='auto', eta='auto', \
                           iterations=iterations, num_topics=num_topics, \
                           passes=passes, eval_every=eval_every)


    vis = gensimvis.prepare(model, corpus, dictionary)
    st.balloons()

    pyLDAvis.save_html(vis, 'lda.html')

    with open('./lda.html', 'r') as f:
        html_string = f.read()
    components.v1.html(html_string, width=1300, height=800, scrolling=False)

    st.write("What do we see here? \n The left panel, labeld Intertopic Distance Map, circles represent different topics and the distance between them. \
        Similar topics appear closer and the dissimilar topics farther. \
        The relative size of a topic's circle in the plot corresponds to the relative frequency of the topic in the corpus. \
        An individual topic may be selected for closer scrutiny by clicking on its circle, or entering its number in the 'selected topic' box in the upper-left.\
        The right panel, include the bar chart of the top 30 terms. When no topic is selected in the plot on the left, the bar chart shows the top-30 most 'salient' terms in the corpus.\
         A term's saliency is a measure of both how frequent the term is in the corpus and how 'distinctive' it is in distinguishing between different topics. \
         Selecting each topic on the right, modifies the bar chart to show the 'relevant' terms for the selected topic. \
         Relevence is defined as in footer 2 and can be tuned by parameter  λ, smaller  λ gives higher weight to the term's distinctiveness while larger λs corresponds to probablity of the term occurance per topics.\
        Therefore, to get a better sense of terms per topic we'll use  λ=0.")

    st.subheader('Top terms by topic')
    topic_summaries = []
    st.write((u'{:20} {}'.format(u'term', u'frequency') + u'\n'))
    for i in range(num_topics):
        st.write(('Topic '+str(i)+' |---------------------\n'))
        tmp = explore_topic(model,topic_number=i, topn=10, output=True )
        topic_summaries += [tmp[:5]]

    st.subheader('Compute Perplexity')
    st.write('Perplexity: '+str(model.log_perplexity(corpus)))


    st.subheader('Getting probabilities from model and appending to dataframe now')

    data['topics'] = model.get_document_topics(corpus)

    sf = pd.DataFrame(data=data['topics'])
    af = pd.DataFrame()

    for i in range(5):
        af[str(i)]=[]

    frames = [sf,af]
    af = pd.concat(frames).fillna(0)

    for i in range(14985):
        for j in range(len(data['topics'][i])):
            af[str(data['topics'][i][j][0])].loc[i] = data['topics'][i][j][1]

    lda_topics = model.show_topics(num_words=5)

    topics = []
    filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]

    for topic in lda_topics:
        topics.append(preprocess_string(topic[1], filters))

    new_topics = []
    for i in topics:
        new_topics.append('_'.join(i))

    af = af.drop(['topics'],axis=1)
    af.columns = new_topics

    new_data = data.join(af)
    new_data = new_data.drop(['topics'],axis=1)


    @st.cache
    def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return new_data.to_csv().encode('utf-8')

    csv = convert_df(new_data)

    st.download_button(
     label="Download data as CSV",
     data=csv,
     file_name='data.csv',
     mime='text/csv')
