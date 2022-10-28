import numpy as np
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize
from collections import defaultdict

import spacy
import neuralcoref




def construct_tfidf_sim_graph_by_gensim(corpus, sim_threshold):
    """Constuct TFIDF similarity graph by Gensim package"""

    sim_graph = []
    #raw_corpus = [' '.join(para) for para in corpus]
    raw_corpus = corpus

    # create English stop words list
    stoplist = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in para.lower().split() if word not in stoplist]
             for para in raw_corpus]
    # Create a set of frequent words
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # stem each word
    processed_corpus = [[p_stemmer.stem(token) for token in text if frequency[token] > 1] for text in texts]

    dictionary = corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    # train the model
    tfidf = models.TfidfModel(bow_corpus)
    # transform the "system minors" string
    corpus_tfidf = tfidf[bow_corpus]

    for i, cor in enumerate(corpus_tfidf):
        if len(cor) == 0:
            print("The corpus_tfidf[i] is None: %s" % str(corpus_tfidf[i]))
            print(bow_corpus[i])
            # exit(1)

    index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary))

    total = 0.
    count_large = 0.
    for i in range(len(corpus_tfidf)):
        sim = index[corpus_tfidf[i]]
        assert len(sim) == len(corpus_tfidf), "the tfidf sim is not correct!"
        sim_graph.append(sim.tolist())

        for s in sim:
            total += 1
            if s > sim_threshold:
                count_large += 1

    print("sim_graph[0]: %s" % str(sim_graph[0]))

    return sim_graph, count_large, total


def get_optimal_ldamodel_by_coherence_values(corpus, texts, dictionary, stop=100, start=10, step=10):
    """
    get the lsi model with optimal number of topics
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LDA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    num_lists = range(start, stop, step)
    for num_topics in num_lists:
        # generate LDA model
        #print(num_topics)
        model = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                                alpha='auto', eta='auto', eval_every=None)  # train model
        model_list.append(model)
        coherencemodel = models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    print("num_topics: %s" % str(num_lists))
    print("coherence_values: %s" % str(coherence_values))

    max_ind = np.argmax(np.array(coherence_values))
    print("opt_num_topics: %s" % num_lists[max_ind])
    return model_list[max_ind]


def construct_lda_sim_graph(corpus, sim_threshold):
    """
    compute lda vector similarity between paragraphs
    :param corpus:
    :param args:
    :return:
    """
    sim_graph = []
    raw_corpus = [' '.join(para) for para in corpus]

    # create English stop words list
    stoplist = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in para.lower().split() if word not in stoplist]
             for para in raw_corpus]
    # Create a set of frequent words
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # stem each word
    processed_corpus = [[p_stemmer.stem(token) for token in text] for text in texts]

    dictionary = corpora.Dictionary(processed_corpus)

    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    # train the model
    #lda = get_optimal_ldamodel_by_coherence_values(corpus=bow_corpus, texts=processed_corpus, dictionary=dictionary, stop=max(11,len(dictionary)))
    lda = models.LdaModel(corpus=bow_corpus, num_topics=10, id2word=dictionary,
                                alpha='auto', eta='auto', eval_every=None, minimum_probability=0.0)

    corpus_lda = lda[bow_corpus]  # create a double wrapper over the original corpus: bow->lda
    index = similarities.MatrixSimilarity(corpus_lda, num_features=len(dictionary))

    print("corpus_lda[0]: %s" % str(corpus_lda[0]))

    total = 0.
    count_large = 0.
    for i in range(len(corpus_lda)):
        sim = index[corpus_lda[i]]

        assert len(sim) == len(corpus_lda), "the lda sim is not correct!"
        sim_graph.append(sim.tolist())

        for s in sim:
            total += 1
            if s > sim_threshold:
                count_large += 1

    print("sim_graph[0]: %s" % str(sim_graph[0]))
    return sim_graph, count_large, total




def construct_coreference_graph(corpus):
    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)
    adj = np.diag([1]*len(corpus))
    lens = []
    for sent in corpus:
        doc = nlp(sent)
        if len(lens) == 0:
            lens.append(len(doc))
        else:
            lens.append(len(doc)+lens[-1])
    text = ' '.join(corpus)
    doc = nlp(text)

    #print(len(doc), lens)
    #print(doc._.coref_clusters)
    for cluster in doc._.coref_clusters:
        inds = []
        for mention in cluster.mentions:
            #print(mention.start)
            for i, l in enumerate(lens):
                if l < mention.start:
                    continue
                inds.append(i-1)
                break
        #print(inds)
        for i in range(len(inds)-1):
            for j in range(i+1, len(inds)):
                adj[inds[i],inds[j]] = 1
                adj[inds[j],inds[i]] = 1
    
    return adj.tolist()
        
    


def get_corpus(infile):
    with open(infile, 'r', encoding='utf-8') as fin:
        corpus = []
        for line in fin:
            corpus.append(eval(line.strip())[:20])
    return corpus


def build_graph(infile, outfile):
    inset = get_corpus(infile)
    print(len(inset))
    with open(outfile,'a') as fout:
        count = 0
        for c in inset:
            #if count < 43722:#135087
            #    count += 1
            #    continue
            adj_sem, count_large, total = construct_tfidf_sim_graph_by_gensim(c, 0.5)
            adj_top, count_large, total = construct_lda_sim_graph(c, 0.5)
            #print(sim_graph)
            #print(count_large)
            #print(total)
            adj_cor = construct_coreference_graph(c)

            for i in range(len(adj_sem)):
                for j in range(len(adj_sem)):
                    adj_sem[i][j] = round(adj_sem[i][j], 2)
            
            for i in range(len(adj_top)):
                for j in range(len(adj_top)):
                    adj_top[i][j] = round(adj_top[i][j], 2)

            adj = []
            adj.append(adj_sem)
            adj.append(adj_top)
            adj.append(adj_cor)
            print(adj)
            fout.write(str(adj) + '\n')

if __name__ == "__main__":
    build_graph('../wikihow/train.src.str', '../wikihow/train.adj')
    #build_graph('../pubmedqa/valid.src.str', '../pubmedqa/valid.adj')
    build_graph('../wikihow/test.src.str', '../wikihow/test.adj')


