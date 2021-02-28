import pandas as pd
from nltk.tokenize import TweetTokenizer
import re,html
import numpy as np
import collections
import pickle

def convert_emojies(t):
    ANGER = "anger"
    FEAR= "fear"
    JOY = "joy"
    SADNESS="sadness"
    anger_set = ['💢','💥','👎','🖕','😤','😠','😡','😾','😧']
    fear_set  = ['👻','☠','💀','😨','😰','😧','😦',
                 '😱','🙀','😿','💉','😵']
    joy_set   = ['✌','👆','👍','👏','🙌','⭐','🎄','🎂','🎁',
                 '🎅','💞','💕','💛','💚','💙','💜','💓','💗',
                 '💖','💘','💝','🎉','👧','👦','👪','💯','😺',
                 '😸','😹','😻','😽','😍','😍','😘','😗','😙','😀',
                 '😂','😁','😆','🙂','😄','😃','☺','😉','😊',
                 '🤗','😋','😝','😙','😚']
    sadness_set = ['💔','💧','😞','😔','😭','😢','😟','😕','🙁',
                   '☹','😖','😫','😩','😓','😪','😿','😷','😿',
                   '🤕','🤒','🤧','🙇']

    t_new = [ ANGER if x in anger_set else x for x in t ]
    t_new = [ FEAR  if x in fear_set else x for x in t_new]
    t_new = [ JOY   if x in joy_set else x for x in t_new]
    t_new = [ SADNESS if x in sadness_set else x for x in t_new]
    return t_new 

def remove_hash_sign(t):
    t_new = [x.replace('#','') for x in t]
    return t_new

def collapse_multiples(t):
    t_new = [re.sub(r'(.)\1+', r'\1\1',x) for x in t]
    return t_new 

def convert_numbers(t):
    t_new = []
    for x in t:
        try:
            x_new = float(x)
            x_new = '<number>'
        except:
            x_new = x
        t_new.append(x_new)
    return t_new

def convert_URL(text):
    text = re.sub(r"\[link\](.*?)\[/link\]","<URL>",text)
    return text

def small_fixup(t):
    new_t = []
    for x in t:
        x_a = x.replace('i\'m','i am').replace('it\'s','it is').replace('don\'t','do not').replace(
            'can\'t','cannot').replace('i\'ve','i have').replace('you\'re','you are').replace(
            'didn\'t','did not').replace('doesn\'t','does not').replace('isn\'t','is not').replace(
            'they\'re','they are').replace('wasn\'t','was not').replace('wouldn\'t','would not').replace(
            'won\'t','will not').replace('it\'ll','it will').replace('we\'re','we are').replace(
            'aren\'t','are not').replace('he\'ll','he will').replace('hasn\'t','has not').replace(
            'she\'ll','she will').replace('we\'ll','we will')
        if x_a == x:
            new_t.append(x_a)
        else:
            xs = x_a.split(' ')
            new_t = new_t + xs
    return new_t

def fixup(x):
    re1 = re.compile(r'  +')
    x = convert_URL(x)
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def get_texts(df,is_test,preserve_case):
    tokenizer = TweetTokenizer(preserve_case=preserve_case,strip_handles=True)
    texts  = df.iloc[:,1].values.astype(np.str)
    if is_test:
        labels = df.iloc[:,3].values
    else:
        labels = df.iloc[:,3].values.astype(np.float)
    
    texts  = [fixup(x) for x in texts]
    tok = [small_fixup(
        convert_numbers(
            remove_hash_sign(
                collapse_multiples(
                    convert_emojies(
                        tokenizer.tokenize(x)))))) for x in texts]    
    return tok, list(labels)

def get_all(df,preserve_case,is_test=False):
    tok,labels = get_texts(df,is_test,preserve_case)    
    return tok, labels

def get_data(args):
    df_trn = pd.read_table(args.data_file+'/train'+'/EI-reg-En-{}-train.txt'.format(args.emotion))
    df_dev = pd.read_table(args.data_file+'/dev'+'/2018-EI-reg-En-{}-dev.txt'.format(args.emotion))
    df_test = pd.read_table(args.data_file+'/test'+'/2018-EI-reg-En-{}-test.txt'.format(args.emotion))
    tok_trn,label_trn   = get_all(df_trn,not args.lowercase) # 1701
    tok_dev,label_dev   = get_all(df_dev,not args.lowercase) # 388
    tok_test,label_test = get_all(df_test,not args.lowercase,True)
    return tok_trn,label_trn,tok_dev,label_dev,tok_test,label_test

def get_natural_data(tok_natural,lowercase):
    res = [small_fixup(
        convert_numbers(
            remove_hash_sign(
                collapse_multiples(
                    convert_emojies(
                        tweet))))) for tweet in tok_natural]
    if lowercase:
        res = [[ word.lower() for word in tweet ]for tweet in res ]

    fake_label_natural = [i for i in range(len(res))]
    return res,fake_label_natural

def get_dictionary(tok_trn,args,extra_tokens,extra_phrases):
    freq = collections.Counter(p for o in tok_trn for p in o)
    itos = [o for o,c in freq.most_common(args.max_vocab) if c>args.min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    itos = add_ecc_words_to_dict(itos,extra_tokens,extra_phrases,args.lowercase)
    pickle.dump(itos,open(args.model_vocab_dir,'wb'))
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    return itos,stoi

def add_ecc_words_to_dict(itos,extra_tokens,extra_phrases,lcase):
    itos.insert(len(itos),'enraged')
    itos.insert(len(itos),'irritating')
    itos.insert(len(itos),'vexing')
    itos.insert(len(itos),'outrageous')
    itos.insert(len(itos),'displeasing')
    for name in extra_tokens:
        if lcase:
            name = name.lower()
        if name not in itos:
            itos.insert(len(itos),name)        
    for noun_p in extra_phrases:
        parts = noun_p.split(' ')
        for part in parts:
            if lcase:
                part = part.lower()
            if part not in itos:
                itos.insert(len(itos),part)
    return itos
    
