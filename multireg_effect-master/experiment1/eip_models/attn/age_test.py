from analyze import *

def check_pretrained_vectors(words,all_data):
    valid_data = [] 
    counter = 0 
    for name in all_data:
        try:
            check_if_exist = words[name] 
            valid_data.append(name)
        except:
            print("the name {} not exist in the word2vec dictionary".format(name))
            counter += 1
    print("out of {} names {} not exist".format(len(all_data),counter))
    return valid_data



def change_names(data,prev,new):
    map_names = dict(zip(prev, new))
    data['Sentence'] = data['Sentence'].apply(lambda x: ' '.join(map_names.get(y, y) for y in x.split()))
    data['Person'] = data['Person'].apply(lambda x: ' '.join(map_names.get(y, y) for y in x.split()))
    return data
