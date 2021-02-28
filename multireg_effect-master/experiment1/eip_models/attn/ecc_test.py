from scipy import stats
from statistics import mean, stdev
from math import sqrt
from ecc_constants import *
from preprocess import *
from model import *

def cohens_d(a0,a1):
    return (mean(a0) - mean(a1)) / (sqrt((stdev(a0) ** 2 + stdev(a1) ** 2) / 2))

def unzip_values(arr1,arr2):
    option_1 = [s[0] for t_e_pair in arr1 for s in t_e_pair] + [s[0] for t_e_pair in arr2 for s in t_e_pair]
    option_2 = [s[1] for t_e_pair in arr1 for s in t_e_pair] + [s[1] for t_e_pair in arr2 for s in t_e_pair]
    return option_1,option_2

def apply_ecc_test(model,stoi,gr_ds,data,emotion_words,device,args):
    _,predictions = generate(args.emotion,model,gr_ds,False,device)
    merged = [ data[i]+(predictions[i],) for i in range(len(predictions))]
    gender_scores_wo_emotion,gender_scores_wi_emotion,race_scores_wo_emotion,race_scores_wi_emotion = race_gender_score(emotion_words,merged,male_names,female_names,male_noun_phrases,female_noun_phrases,af_names,eu_names)

    g_m,g_f = unzip_values(gender_scores_wi_emotion,gender_scores_wo_emotion)
    r_af,r_eu = unzip_values(race_scores_wi_emotion,race_scores_wo_emotion)
    sg,pg = stats.ttest_rel(g_m, g_f)  # 429
    sr,pr = stats.ttest_rel(r_af,r_eu) # 39
    cdg = cohens_d(g_m,g_f)
    cdr = cohens_d(r_af,r_eu)
    print("cohen's d [gender]/[race]:{:.6f}/{:.6f}".format(cdg,cdr))
    print("[gender] paired 2-test s:{},p:{}".format(sg,pg))
    print("[race]   paired 2-test s:{},p:{}".format(sr,pr))

def get_gb_data(df_gb,emotion):
    result = []
    for index,row in df_gb.loc[df_gb['Emotion']==emotion].iterrows():
        result.append((row['ID'],row['Sentence'],row['Template'],
                       row['Person'],row['Gender'],row['Race'],row['Emotion'],row['Emotion word']))
    for index,row in df_gb.loc[df_gb['Emotion'].isnull()].iterrows():
        result.append((row['ID'],row['Sentence'],row['Template'],
                       row['Person'],row['Gender'],row['Race'],row['Emotion'],row['Emotion word']))
    return result

def get_ecc_texts(data,preserve_case):
    tokenizer = TweetTokenizer(preserve_case=preserve_case,strip_handles=True)
    tok = [(x[0],x[1],
            small_fixup(
                convert_numbers(
                    remove_hash_sign(
                        collapse_multiples(
                            convert_emojies(
                                tokenizer.tokenize(fixup(x[1]))))))),x[2],x[3],x[4],x[5],x[6],x[7]
    )for x in data]
    tok2 = [x[2] for x in tok]
    return tok2



def race_gender_score(emotion_words,merged,male_names,female_names,male_noun_phrases,female_noun_phrases,af_names,eu_names):
    # 1584 gender score pairs
    gender_scores_wi_emotion = [] 
    gender_scores_wo_emotion = []
    # 144 race score pairs 
    race_scores_wi_emotion = []
    race_scores_wo_emotion = []
    for template in templates_wi_emotion:
        for emotion_word in emotion_words:
            instances = list(filter(lambda x:x[2]==template and x[7]==emotion_word,merged))
            if len(instances) == 0:
                continue
            #print(template," ",emotion_word)
            score_gender = get_gender_scores(instances,male_names,female_names,male_noun_phrases,female_noun_phrases)
            score_race   = get_race_scores(instances,af_names,eu_names)   # There is a 1 score here  for that templat emotion word
            gender_scores_wi_emotion.append(score_gender)
            race_scores_wi_emotion.append(score_race)
            
    for template in templates_wo_emotion:
        instances = list(filter(lambda x:x[2] == template,merged))
        score_gender = get_gender_scores(instances,male_names,female_names,male_noun_phrases,female_noun_phrases)
        
        score_race   = get_race_scores(instances,af_names,eu_names)
        gender_scores_wo_emotion.append(score_gender) 
        race_scores_wo_emotion.append(score_race)
    return gender_scores_wo_emotion,gender_scores_wi_emotion,race_scores_wo_emotion,race_scores_wi_emotion


def get_gender_scores(data,male_names,female_names,male_noun_phrases,female_noun_phrases):
    # Proper name score
    m = list(filter(lambda x:x[3] in male_names,data))
    f = list(filter(lambda x:x[3] in female_names,data))
    ms = sum([x[-1] for x in m])/len(m) # male avg score
    fs = sum([x[-1] for x in f])/len(f) # female avg score
    # Noun phrase scores
    mnp = male_noun_phrases
    fnp = female_noun_phrases    
    np_scores = [] 
    for i in range(len(mnp)):
        c1 = False
        c2 = False
        for d in data:
            if d[3] == mnp[i]:
                s1 = d[-1]
                c1 = True
            elif d[3] == fnp[i]:
                s2 = d[-1]
                c2 = True
            else:
                continue
        if c1 == True and c2 == True:
            np_scores.append((s1,s2))
    return [(ms,fs)] + np_scores

def get_race_scores(data,af_names,eu_names):
    af = list(filter(lambda x:x[3] in af_names,data))
    eu = list(filter(lambda x:x[3] in eu_names,data)) 
    afs = sum([x[-1] for x in af])/len(af) # male avg score
    eus = sum([x[-1] for x in eu])/len(eu) # female avg score
    return [(afs,eus)]


def apply_ecc_test_age(model,stoi,age_ds,data,emotion_words,old_names,young_names,device,args):
    _,predictions = generate(args.emotion,model,age_ds,False,device)
    merged = [ data[i]+(predictions[i],) for i in range(len(predictions))]
    age_scores_wi_emotion,age_scores_wo_emotion = age_score(emotion_words,merged,old_names,young_names)    
    # Split race outputs 
    age_old,age_young = unzip_values(age_scores_wi_emotion,age_scores_wo_emotion)
    # Tests 
    sa,pa = stats.ttest_rel(age_old,age_young)  # 429
    cda = cohens_d(age_old,age_young)
    print("cohen's d [age]:{:.6f}".format(cda))
    print("[age] paired 2-test s:{},p:{}".format(sa,pa))

def age_score(emotion_words,merged,old_names,young_names):
    age_scores_wi_emotion = []
    age_scores_wo_emotion = []
    for template in templates_wi_emotion:
        for emotion_word in emotion_words:
            instances = list(filter(lambda x:x[2]==template and x[7]==emotion_word,merged))
            if len(instances) == 0:
                continue
            score_age   = get_age_scores(instances,old_names,young_names)
            age_scores_wi_emotion.append(score_age)
    for template in templates_wo_emotion:
        instances = list(filter(lambda x:x[2] == template,merged))
        score_age   = get_age_scores(instances,old_names,young_names)
        age_scores_wo_emotion.append(score_age)        
    return age_scores_wi_emotion,age_scores_wo_emotion


def get_age_scores(data,old_names,young_names):
    old = list(filter(lambda x:x[3] in old_names,data))
    young = list(filter(lambda x:x[3] in young_names,data)) 
    olds = sum([x[-1] for x in old])/len(old) # old avg score
    youngs = sum([x[-1] for x in young])/len(young) # young avg score
    return [(olds,youngs)]
