import os
import random
import logging,argparse
import gensim.models.keyedvectors as word2vec
from scipy.stats import pearsonr
from preprocess import *
from ecc_test import *
from age_test import *
from model import *

GPU_ID = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
torch.cuda.set_device(GPU_ID)
device = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

_seed=4321
torch.manual_seed(_seed)
np.random.seed(_seed)
torch.cuda.manual_seed(_seed)
random.seed(_seed)


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
    parser.add_argument("--logfile", type=str,help="name of the logfile",default='./log_cnn.out')
    parser.add_argument("--data_file", type=str,default='../data/Ereg')
    parser.add_argument("--w2vecpath",type=str,default='../data/embeddings/GoogleNews-vectors-negative300.txt')
    parser.add_argument("--embedding_file", type=str,help="pre-trained embeddings",
                        default='../data/embeddings/glove.twitter.27B.200d.txt')
    parser.add_argument("--best_model_save_dir", type=str,help="best model path",
                        default='./best_model_rewrite.pth')
    parser.add_argument("--model_vocab_dir", type=str,help="vocab dir",default='./itos_rewwrite.pkl')
    parser.add_argument("--saved_fast_embedding_file",type=str,help="to escape from loading word2vec each time",
                        default='./rewrite_embedding.pkl')
    parser.add_argument("--gender_bias_dataset",type=str,help="path for ecc dataset",
                        default='../data/Equity-Evaluation-Corpus/Equity-Evaluation-Corpus.csv'.format(root_path))
    parser.add_argument("--emotion", type=str,help="emotion type",default='anger')
    parser.add_argument("--bs", type=int,help="batch size",default=64)
    parser.add_argument("--es", type=int,help="embedding size",default=300)    
    parser.add_argument("--hs", type=int,help="hidden size",default=200)
    parser.add_argument("--layer_num", type=int,help="number of layers in lstm",default=1)
    parser.add_argument("--filter_number", type=int,help="number of filters for cnn",default=200)
    parser.add_argument("--dense_out", type=int,help="dimension of dense layer",default=30)
    parser.add_argument("--dp", type=float,help="dropout",default=0.5)
    parser.add_argument("--epoch", type=int,help="epoch",default=10)    
    parser.add_argument("--min_freq", type=int,help="min frequency to add token to vocabulary",default=0)
    parser.add_argument("--max_vocab", type=int,help="max len for vocabulary",default=60000)
    parser.add_argument("--max_len", type=int,help="max len for tweet",default=80)
    parser.add_argument("--n_gram_num", type=int,help="number_of_ngrams",default=5)
    parser.add_argument("--shuffle",help="shuffle data or not",action='store_true')
    parser.add_argument("--lowercase",help="lowercase",action='store_true')
    parser.add_argument("--test_set",help="use test set",action='store_true')
    parser.add_argument("--ecc_set",help="use census set",action='store_true')
    parser.add_argument("--census_set",help="use census set",action='store_true')
    parser.add_argument("--regression_analysis",help="do regression",action='store_true')

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
    tok_trn,pos_trn,label_trn,tok_dev,pos_dev,label_dev,tok_test,pos_test,label_test = get_ark_tweet_data(args)
    lex_trn,lex_dev,lex_tst = get_sentence_lexicons(args)
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
    else:
        words = word2vec.KeyedVectors.load_word2vec_format(args.w2vecpath, binary=False)
        valid_old_names     = check_pretrained_vectors(words,all_old_names)
        valid_young_names   = check_pretrained_vectors(words,all_young_names)
        valid_similar_names = check_pretrained_vectors(words,all_similar_names)
        extra_tokens = male_names + female_names + valid_old_names + valid_young_names+valid_similar_names
        extra_phrases = male_noun_phrases + female_noun_phrases
        itos,stoi  = get_dictionary(tok_trn,args,extra_tokens,extra_phrases)
        embeddings = get_embedding_matrix(args,itos,words)
        saved_dict = {}
        saved_dict['valid_old_names'] = valid_old_names
        saved_dict['valid_young_names'] = valid_young_names
        saved_dict['valid_similar_names'] = valid_similar_names
        saved_dict['embeddings'] = embeddings
        pickle.dump(saved_dict,open(args.saved_fast_embedding_file,'wb'))
       
    ptoi = get_pos_dictionary(pos_trn,args)
    print(ptoi)
    print(len(stoi))
    print(len(ptoi))
    # this name_vecs will be used in regression analysis below 
    name_vecs  = {}
    for name in valid_old_names + valid_young_names+valid_similar_names:
        name_vecs[name] = embeddings[stoi[name],:]
        
    print("embeddings has been read")
    trn  = np.array([[stoi[o] for o in p] for p in tok_trn])
    dev  = np.array([[stoi[o] for o in p] for p in tok_dev])
    test = np.array([[stoi[o] for o in p] for p in tok_test])

    trn_pos  = np.array([[ptoi[o] for o in p] for p in pos_trn])
    dev_pos  = np.array([[ptoi[o] for o in p] for p in pos_dev])
    test_pos = np.array([[ptoi[o] for o in p] for p in pos_test])    
    
    loss_function = nn.MSELoss() # take sqrt to use it as RMSE
    model = Anger2(embeddings,args.filter_number,args.dense_out,args.n_gram_num,len(ptoi))
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    trn_batches = make_batch(trn,label_trn,lex_trn,trn_pos,args.bs,args.max_len,stoi,
                             ptoi,is_shuffle=True)
    dev_batches = make_batch(dev,label_dev,lex_dev,dev_pos,args.bs,args.max_len,stoi,
                             ptoi,is_shuffle=True)
    print("trn/dev batch number:{}/{}".format(len(trn_batches),len(dev_batches)))

    model,optimizer = main_loop(model,optimizer,trn_batches,dev_batches,loss_function,device,args)
    
    # GENERATION AND PEARSON CORRELATION EVALUATION
    trn_batches = make_batch(trn,label_trn,lex_trn,trn_pos,1,args.max_len,stoi,
                             ptoi,is_shuffle=False,is_gen=True)
    dev_batches = make_batch(dev,label_dev,lex_dev,dev_pos,1,args.max_len,stoi,
                             ptoi,is_shuffle=False,is_gen=True)

    golds,predictions = generate(args.emotion,model,dev_batches,label_dev,device)
    corr, p_value = pearsonr(golds, predictions)
    print("[last model] correlation/p_value: {:.3f}/{:.10f}".format(corr,p_value))

    # With lowest dev-set error model
    model = Anger2(embeddings,args.filter_number,args.dense_out,args.n_gram_num,len(ptoi))
    model.load_state_dict(torch.load(args.best_model_save_dir))
    model.to(device)
    golds,predictions = generate(args.emotion,model,dev_batches,label_dev,device)
    corr, p_value = pearsonr(golds,predictions)
    print("[best model] correlation/p_value: {:.3f}/{:.10f}".format(corr,p_value))

    if args.test_set:
        # For Test set
        fake_label = [i for i in range(len(test))]
        test_batches = make_batch(test,fake_label,lex_tst,test_pos,1,args.max_len,stoi,ptoi,is_shuffle=False,is_gen=True,is_test=True)
        golds,predictions = generate(args.emotion,model,test_batches,label_test,device)
    if args.ecc_set:
        # For ECC  data - paired t-test
        df_gb = pd.read_csv(args.gender_bias_dataset)
        data = get_gb_data(df_gb,args.emotion)
        tok = get_ecc_texts(data,not args.lowercase)
        tok = np.array([[stoi[o] for o in p] for p in tok])
        fake_label = [i for i in range(len(tok))]
        gr_ds = make_batch(tok,fake_label,1,args.max_len,stoi,is_shuffle=False,is_gen=True,is_test=True)
        emotion_words = list(df_gb.loc[df_gb['Emotion']==args.emotion]['Emotion word'].unique())
        apply_ecc_test(model,stoi,gr_ds,data,emotion_words,device,args)

    if args.census_set:
        # FOR CENSUS Data - paired t-test
        name_sets = get_name_sets(valid_old_names,valid_young_names,20)
        for (old_names,young_names) in name_sets:
            df_age = pd.read_csv(args.gender_bias_dataset)
            df_age = change_names(df_age,af_names,old_names)
            df_age = change_names(df_age,eu_names,young_names)
            data_age = get_gb_data(df_age,args.emotion)
            tok_age = get_ecc_texts(data_age,not args.lowercase)
            tok_age = np.array([[stoi[o] for o in p] for p in tok_age])
            fake_label = [i for i in range(len(tok_age))]
            age_ds = make_batch(tok_age,fake_label,1,args.max_len,stoi,is_shuffle=False,is_gen=True,is_test=True)
            emotion_words_age = list(df_age.loc[df_age['Emotion']==args.emotion]['Emotion word'].unique())
            apply_ecc_test_age(model,stoi,age_ds,data_age,emotion_words_age,old_names,young_names,device,args)


def get_name_sets(set1,set2,n):
    random.shuffle(set1)
    random.shuffle(set2)
    return [(set1[:n],set2[:n])]


if __name__=='__main__':
    main()
