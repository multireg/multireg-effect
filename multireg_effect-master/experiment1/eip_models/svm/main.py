import os
import random
import logging,argparse
import gensim.models.keyedvectors as word2vec
from scipy.stats import pearsonr
from sklearn.svm import SVR
from preprocess import *
from ecc_test import *
from age_test import *

GPU_ID = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

_seed=4321
np.random.seed(_seed)
random.seed(_seed)

def get_input_matrix(data,vs):
    res = np.zeros((len(data),vs))
    for ix,d in enumerate(data):
        res[ix,d] = 1
    return res

def get_embedding_matrix(args,itos,words):
    vs = len(itos)
    embeddings = np.zeros((vs, args.es), dtype=np.float32)
    for i,t in enumerate(itos):
        if t == "_pad_":
            continue
        else:
            try:
                e = words[t]
            except:
                e = np.random.uniform(-0.25,0.25,300)
        embeddings[i,:] = e
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str,help="name of the logfile",default='./log_svm.out')
    parser.add_argument("--data_file", type=str,default='../data/Ereg')
    parser.add_argument("--w2vecpath", type=str,help="pre-trained embeddings",
                        default='../data/embeddings/glove.twitter.27B.200d.txt')
    parser.add_argument("--best_model_save_dir", type=str,help="best model path",
                        default='./best_model.pth')    
    parser.add_argument("--model_vocab_dir", type=str,help="vocab dir",default='./itos_rewrite.pkl')
    parser.add_argument("--saved_fast_embedding_file",type=str,help="to escape from loading word2vec each time",
                        default='./rewrite_embedding.pkl')
    parser.add_argument("--gender_bias_dataset",type=str,help="path for ecc dataset",
                        default='../data/Equity-Evaluation-Corpus/Equity-Evaluation-Corpus.csv')
    parser.add_argument("--emotion", type=str,help="emotion type",default='anger')
    parser.add_argument("--bs", type=int,help="batch size",default=8)
    parser.add_argument("--es", type=int,help="embedding size",default=300)    
    parser.add_argument("--hs", type=int,help="hidden size",default=200)
    parser.add_argument("--layer_num", type=int,help="number of layers in lstm",default=1)
    parser.add_argument("--filter_number", type=int,help="number of filters for cnn",default=200)
    parser.add_argument("--dense_out", type=int,help="dimension of dense layer",default=200)
    parser.add_argument("--dp", type=float,help="dropout",default=0.2)
    parser.add_argument("--epoch", type=int,help="epoch",default=10)    
    parser.add_argument("--min_freq", type=int,help="min frequency to add token to vocabulary",default=0)
    parser.add_argument("--max_vocab", type=int,help="max len for vocabulary",default=60000)
    parser.add_argument("--max_len", type=int,help="max len for tweet",default=50)
    parser.add_argument("--n_gram_num", type=int,help="number_of_ngrams",default=4)
    parser.add_argument("--shuffle",help="shuffle data or not",action='store_true')
    parser.add_argument("--lowercase",help="lowercase",action='store_true')
    parser.add_argument("--bidirectional",help="lowercase",action='store_false')    

    args = parser.parse_args()
    options = vars(args)
    print(options)
    # LOGGER 
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(args.logfile)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(options)

    # DATA PROCESSING
    tok_trn,label_trn,tok_dev,label_dev,tok_test,label_test = get_data(args)
    #words = word2vec.KeyedVectors.load_word2vec_format(args.w2vecpath, binary=False)

    # Dictionary
    if os.path.isfile(args.saved_fast_embedding_file):
        saved_dict = pickle.load(open(args.saved_fast_embedding_file,'rb'))
        valid_old_names = saved_dict['valid_old_names']
        valid_young_names = saved_dict['valid_young_names']
        valid_similar_names = saved_dict['valid_similar_names']
        embeddings = saved_dict['embeddings']
        extra_tokens = male_names + female_names + valid_old_names + valid_young_names+valid_similar_names
        extra_phrases = male_noun_phrases + female_noun_phrases
        itos,stoi  = get_dictionary(tok_trn,args,extra_tokens,extra_phrases)
        vs = len(stoi)
    else:
        print("read word2vecs....")
        words = word2vec.KeyedVectors.load_word2vec_format(args.w2vecpath, binary=False)
        valid_old_names     = check_pretrained_vectors(words,all_old_names)
        valid_young_names   = check_pretrained_vectors(words,all_young_names)
        valid_similar_names = check_pretrained_vectors(words,all_similar_names)
        extra_tokens = male_names + female_names + valid_old_names + valid_young_names+valid_similar_names
        extra_phrases = male_noun_phrases + female_noun_phrases
        itos,stoi  = get_dictionary(tok_trn,args,extra_tokens,extra_phrases)
        vs = len(stoi)
        embeddings = get_embedding_matrix(args,itos,words)
        saved_dict = {}
        saved_dict['valid_old_names'] = valid_old_names
        saved_dict['valid_young_names'] = valid_young_names
        saved_dict['valid_similar_names'] = valid_similar_names
        saved_dict['embeddings'] = embeddings
        pickle.dump(saved_dict,open(args.saved_fast_embedding_file,'wb'))

    # this name_vecs will be used in regression analysis below 
    
    name_vecs  = {}
    for name in valid_old_names + valid_young_names+valid_similar_names+male_names+female_names:
        name_vecs[name] = embeddings[stoi[name],:]
        
    print("embeddings has been read")
    trn  = np.array([[stoi[o] for o in p] for p in tok_trn])
    dev  = np.array([[stoi[o] for o in p] for p in tok_dev])
    test = np.array([[stoi[o] for o in p] for p in tok_test])
    print("vocabulary size:",vs)


    trnX  = get_input_matrix(trn,vs)
    devX  = get_input_matrix(dev,vs)
    testX = get_input_matrix(test,vs)

    svr_poly = SVR(kernel='linear',C=0.1)
    svr_poly = svr_poly.fit(trnX, label_trn)
    predictions = svr_poly.predict(devX)
    print("prediction length:{}, devset label length:{}".format(len(predictions),len(label_dev)))
    corr, p_value = pearsonr(predictions, label_dev)
    print("Pearson r on dev set:{}".format(corr))
    
    
    # For ECC  data - paired t-test
    df_gb = pd.read_csv(args.gender_bias_dataset)
    data  = get_gb_data(df_gb,args.emotion)
    tok   = get_ecc_texts(data,not args.lowercase)
    tok   = np.array([[stoi[o] for o in p] for p in tok])
    eccX  = get_input_matrix(tok,vs)
    predictions = svr_poly.predict(eccX)
    print("type of predictions:",type(predictions))
    fake_label = [i for i in range(len(tok))]

    emotion_words = list(df_gb.loc[df_gb['Emotion']==args.emotion]['Emotion word'].unique())
    apply_ecc_test(predictions,stoi,data,emotion_words,args)

    # FOR CENSUS Data - paired t-test
    #name_sets = get_name_sets(valid_old_names,valid_young_names,20)
    xx1 = used_old_female_white+used_old_female_black+used_old_male_white+used_old_male_black
    xx2 = used_young_female_white+used_young_female_black+used_young_male_white+used_young_male_black
    name_sets = [(xx1,xx2)]
    for (old_names,young_names) in name_sets:
        df_age = pd.read_csv(args.gender_bias_dataset)
        df_age = change_names(df_age,af_names,old_names)
        df_age = change_names(df_age,eu_names,young_names)
        data_age = get_gb_data(df_age,args.emotion)
        tok_age = get_ecc_texts(data_age,not args.lowercase)
        tok_age = np.array([[stoi[o] for o in p] for p in tok_age])
        censusX = get_input_matrix(tok_age,vs)
        predictions = svr_poly.predict(censusX)
        fake_label = [i for i in range(len(tok_age))]
        emotion_words_age = list(df_age.loc[df_age['Emotion']==args.emotion]['Emotion word'].unique())
        apply_ecc_test_age(predictions,stoi,data_age,emotion_words_age,old_names,young_names,args)


    
def get_name_sets(set1,set2,n):
    random.shuffle(set1)
    random.shuffle(set2)
    return [(set1[:n],set2[:n])]


if __name__=='__main__':
    main()
