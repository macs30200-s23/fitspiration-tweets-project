import sys
sys.path.insert(0, '..')
import utils
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.utils import effective_n_jobs
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models
import pandas as pd
# set the maximum width of columns to None to display the full content
pd.set_option('max_colwidth', None)
import seaborn as sns
sns.set_style("ticks")
sns.set(font_scale=1.2, palette="deep")
# Set the figure size and resolution
fig = plt.figure(figsize=(10, 8), dpi=100)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

'''
Hashtag and TF-IDF analysis
'''

def topn_hashtags(df_col, n, period_name):
    '''
    Input:
        df_col: the column that contains list of hashtags
    '''

    # Concatenate all the hashtags into a single list
    all_hashtags = []
    for ht_list in df_col:
        all_hashtags.extend(eval(ht_list))
    
    # Excluding #fitspo
    hashtags_without_fitspo = [tag for tag in all_hashtags if tag != '#fitspo']

    # Create frequency distribution object
    fd = nltk.FreqDist(hashtags_without_fitspo)

    # Get the top n most frequent hashtags
    top_hashtags = fd.most_common(n)
    # Set the font size
    sns.set(font_scale=1.2)
    plt.rcParams['font.size'] = 12
    
    # Create a horizontal bar chart of the top n most frequent hashtags
    sns.barplot(x=[ht[1] for ht in top_hashtags], y=[ht[0] for ht in top_hashtags], color='grey')
    plt.xlabel('Frequency')
    plt.ylabel('Hashtags')
    plt.title(f'Top {n} Most Common Hashtags in Fitspirational Tweets')
    plt.savefig(f'visualization/{period_name}_top{n}_hashtags.png', dpi=300, bbox_inches='tight')
    plt.show()

    return


def bigram_tfidf(df_col, n, period_name):

    # Create a CountVectorizer with bigram range
    vectorizer = CountVectorizer(ngram_range=(2, 2))

    # Fit the vectorizer on the corpus of processed tweets
    corpus = df_col
    bigram_matrix = vectorizer.fit_transform(corpus)

    # Convert bigram matrix to TF-IDF matrix
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(bigram_matrix)

    # Get feature names and create a dataframe
    feature_names = vectorizer.get_feature_names()
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)


    # Get the mean TF-IDF score for each bigram and sort by score
    mean_scores = df_tfidf.mean().sort_values(ascending=False)

    # Get the top n most significant bigrams
    top_bigrams = mean_scores[:n]
    
    # Set the font size
    sns.set(font_scale=1.2)
    plt.rcParams['font.size'] = 12

    # Create a bar plot of the top salient bigrams
    sns.barplot(x=top_bigrams.values, y=top_bigrams.index, color='grey')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Bigram')
    # plt.title(f'Top {n} Most Significant Bigrams')
    plt.savefig(f'visualization/{period_name}_top{n}_bigrams.png', dpi=300, bbox_inches='tight')
    plt.show()

    return


'''
Pre-processing
'''
# Define lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    '''
    Lemmatize a string
    '''

    tokens = word_tokenize(text)

    # Perform POS tagging
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatize each word based on its POS tag
    lemmas = []
    for token, pos in pos_tags:
        if pos.startswith('J'):
            # Adjective
            lemma = lemmatizer.lemmatize(token, wordnet.ADJ)
        elif pos.startswith('V'):
            # Verb
            lemma = lemmatizer.lemmatize(token, wordnet.VERB)
        elif pos.startswith('N'):
            # Noun
            lemma = lemmatizer.lemmatize(token, wordnet.NOUN)
        elif pos.startswith('R'):
            # Adverb
            lemma = lemmatizer.lemmatize(token, wordnet.ADV)
        else:
            # Use default lemmatization (noun)
            lemma = lemmatizer.lemmatize(token)

        lemmas.append(lemma)
    
    return lemmas


def tweet_per_user(df, period_name):
    '''
    Check num of tweets per user
    '''

    print(df['Username'].value_counts()[:10])
    # Count the number of occurrences of each username
    counts = df['Username'].value_counts()

    # Group the counts by the number of occurrences, and count the number of usernames with that count
    grouped_counts = counts.groupby(counts).count()

    # Plot the histogram
    plt.bar(grouped_counts.index, grouped_counts.values)
    plt.xlabel('Number of Tweets by the Same Author')
    plt.ylabel('Number of Authors')
    plt.savefig(f'visualization/{period_name}_ntweets_per_author.png')
    plt.show();

    return


def make_bigrams(lemmas):
    '''
    Make bigrams for words within a given document
    '''
    bigram = models.Phrases(lemmas, min_count=5)
    bigram_mod = bigram.freeze()
    return [bigram_mod[doc] for doc in lemmas]


def preprocess_lda(df, period_name):
    '''
    Get lemmas and prepare bag-of-words corpus for building lda model
    '''

    df['lemmas'] = df['processed_tweet'].apply(lemmatize_text)
    # inspect the lemmas column 
    print(utils.show_random_5(df, 'lemmas'))

    # concatenate the lemmas of tweets from the same user into one document
    tweet_per_user(df, period_name)
    grouped = df.groupby('Username')['lemmas']\
                .apply(lambda x: [lemma for lemmas in x for lemma in lemmas])\
                .reset_index()

    # lemmas is a list of lists, each list representing a document
    lemmas = grouped['lemmas'].to_list()    
    
    # make bigrams
    lemmas = make_bigrams(lemmas)

    # prepare for lda
    dictionary = corpora.Dictionary(lemmas)
    bow_corpus = [dictionary.doc2bow(doc) for doc in lemmas]

    return lemmas, dictionary, bow_corpus


'''
Build models
'''

# define a list of seed words for each topic
# body idealization/focus on physical appearance
body_idealization = ['fit', 'toned', 'body', 'lean', 'muscular', 'muscle', 'sexy', 
    'ripped', 'shredded', 'defined', 'cut', 'jacked']
# weight stigmatization/focus
weight_stigmatization = ['kg', 'lb', 'pound', 'obese', 'overweight', 'fat', 
            'chubby', 'heavy', 'flabby', 'weight', 'fat', 'weight loss', 'fat loss']
# objectification
objectification = ['hot', 'sexy', 'attractive', 'desirable', 'seductive', 
        'alluring', 'provocative', 'sensual', 'voluptuous', 'lustful', 'erotic']
# restrictive eating
restrictive_eating = ['diet', 'eat', 'clean eating', 'low-carb', 'calorie', 'paleo', 
            'juice cleanse', 'detox', 'fasting', 'calorie counting']
# gender roles/sexualization
gender_roles_sexualization = ['women', 'men', 'woman', 'man', 'masculine', 'feminine', 
        'alpha', 'beta', 'dominant']
# health and fitness information like example workouts
health_fitness_information = ['selfcare', 'workout', 'exercise', 'day',
        'nutrition', 'healthy', 'fitness', 'lifting', 'cardio', 'running', 
        'yoga', 'pilates', 'diet', 'equipment']
# motivation and inspiration
motivation_inspiration = ['motivated', 'motivation', 'inspired', 'drill',
        'ambitious', 'driven', 'goal', 'empowered', 'dedicated', 'persistent', 
        'resilient', 'positive']
# social media ads
ads = ['free', 'shipping', 'worldwide', 'united_states', 'buy', 'get', 
        'delivery', 'link', 'bio', 'merch', 'promo', 'share', 'click']

# according to the literature
topics_1 = [body_idealization, weight_stigmatization, objectification, restrictive_eating, gender_roles_sexualization, health_fitness_information, motivation_inspiration]

# adjusted after the pilot run
topics_2 = [body_idealization, weight_stigmatization, restrictive_eating, health_fitness_information, motivation_inspiration, ads]

# adjusted again
topics_3 = [body_idealization+objectification, weight_stigmatization+restrictive_eating, 
    gender_roles_sexualization, health_fitness_information, 
    motivation_inspiration, ads]

def build_ldamodel(dictionary, bow_corpus, period_name, prior_topic_lst):
    '''
    Build lda model guided by fitspiration-specific knowledge
    '''
    # set a prior probability of 0.1 for the seed words 
    # and 0.9 for the rest of the words in each topic 
    num_topics = len(prior_topic_lst)

    eta = [[0.1 if word in topic else 0.9 for word in dictionary.token2id] 
                                            for topic in prior_topic_lst]
    ldamodel = models.LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary,
                                passes=20, alpha='auto', eta=eta,
                                random_state=42, iterations=400)
    ldamodel.save(f'model/{period_name}_4p20_i400.model')
    return ldamodel


def compute_coherence_value(ldamodel, bow_corpus, dictionary, lemmas):

    # `ldamodel` is the trained LDA model
    # `corpus` is the document-term matrix in BoW format
    # `dictionary` is the gensim Dictionary object

    # Calculate the semantic coherence score using the UMass measure
    coherence_umass = models.CoherenceModel(model=ldamodel, corpus=bow_corpus, 
                dictionary=dictionary, coherence='u_mass').get_coherence()

    # Calculate the topic coherence score using the c_v measure
    coherence_cv = models.CoherenceModel(model=ldamodel, corpus=bow_corpus, 
                dictionary=dictionary, texts=lemmas, coherence='c_v').get_coherence()

    print("Coherence (UMass):", coherence_umass)
    print("Coherence (c_v):", coherence_cv)
    return coherence_umass, coherence_cv
    
    
def analyze_topics(ldamodel, dictionary, bow_corpus, period_name):
    '''
    Analyze each topic with the help of plLDAvis
    '''

    # Print the topics generated by the LDA model
    topics = ldamodel.print_topics(num_words=20)
    print(f"Topics from {period_name} are:")
    for topic in topics:
        print(topic)

    pyLDAvis.enable_notebook()
    # prepare visualization
    vis = pyLDAvis.gensim_models.prepare(ldamodel, bow_corpus, dictionary)
    # save visualization as HTML file
    pyLDAvis.save_html(vis, f'visualization/{period_name}_lda_visualization.html')
    print(f"{period_name} pyLDAvis html saved!")
    
    return


def topn_texts_by_topic(period_name, df, ldamodel, bow_corpus):
    '''
    Write the top texts in a txt file
    '''
    topn_texts_by_topic = {}
    
    with open(f'results/{period_name}_tweets.txt', 'w') as f:
        
        for i in range(len(ldamodel.print_topics())):
            # For each topic, collect the most representative tweet(s)
            # (i.e. highest probability containing words belonging to topic):
            top = sorted(zip(range(len(bow_corpus)), ldamodel[bow_corpus]),
                        reverse=True,
                        key=lambda x: abs(dict(x[1]).get(i, 0.0)))
            topn_texts_by_topic[i] = [j[0] for j in top[:5]]

            # Print out the topn tweets for each topic and return their indices as a
            # dictionary for further analysis:
            f.write("Topic " + str(i))
            f.write("\n")
            f.write(df.iloc[topn_texts_by_topic[i]]['processed_tweet'].to_string())
            f.write("\n")
            f.write("*******************************")
            f.write("\n")
    
    return


def lda_pipeline(df, period_name, prior_topic_lst):
    # preprocess for lda
    lemmas, dictionary, bow_corpus = preprocess_lda(df, period_name)

    # build lda model
    ldamodel = build_ldamodel(dictionary, bow_corpus, period_name, prior_topic_lst)

    # compute coherence score
    compute_coherence_value(ldamodel, bow_corpus, dictionary, lemmas)

    # analyze topics
    analyze_topics(ldamodel, dictionary, bow_corpus, period_name)
    # write most representative tweets per topic
    topn_texts_by_topic(period_name, df, ldamodel, bow_corpus)
    
    return ldamodel, bow_corpus


def fill_topic_weights(df_row, bow_corpus, ldamodel):
    '''
    Fill DataFrame rows with topic weights for topics in songs.

    Modifies DataFrame rows *in place*.
    '''
    try:
        for i in ldamodel[bow_corpus[df_row.name]]:
            df_row[str(i[0])] = i[1]
    except:
        return df_row
    return df_row