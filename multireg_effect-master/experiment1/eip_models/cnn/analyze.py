import pandas as pd 

def sort_data(frame,column_name):
    return frame.sort_values(column_name,ascending=False)

def eleminate_lt_n(frame,column_name,n):
    return frame.loc[frame[column_name] > n]

def ignore_full_prob(frame,column_name):
    return frame.loc[frame[column_name] < 1.0]

def get_most_p_old(file_name,n,lt=0,ignore_full_p=True):
    df = pd.read_csv(file_name,index_col=0)
    df = eleminate_lt_n(df,"old_N",lt)
    if ignore_full_p:
        df = ignore_full_prob(df,"old_p")    
    df_sorted = sort_data(df,"old_p")
    return df_sorted.head(n)

def get_most_p_young(file_name,n,lt=0,ignore_full_p=True):
    df = pd.read_csv(file_name,index_col=0)
    df = eleminate_lt_n(df,"young_N",lt)
    if ignore_full_p:
        df = ignore_full_prob(df,"young_p")
    df_sorted = sort_data(df,"young_p")
    return df_sorted.head(n)

def get_similar_p_values(file_name,lt=0,diff_range=0.01):
    df = pd.read_csv(file_name,index_col=0)
    df = eleminate_lt_n(df,"old_N",lt)
    df = eleminate_lt_n(df,"young_N",lt)    
    return df.loc[abs(df['young_p']-df['old_p']) <diff_range]
    

def get_most_n_old(file_name,n,lt=0):
    df = pd.read_csv(file_name,index_col=0)
    df = eleminate_lt_n(df,"old_N",lt)    
    df_sorted = sort_data(df,"old_N")
    return df_sorted.head(n)

def get_most_n_young(file_name,n,lt=0):
    df = pd.read_csv(file_name,index_col=0)
    df = eleminate_lt_n(df,"young_N",lt)
    df_sorted = sort_data(df,"young_N")
    return df_sorted.head(n)


pd.set_option('display.max_rows', 2100)
file_name = "/home/users2/dayaniey/mardy/erenay/bias/age/result.csv"
most_p_young = get_most_p_young(file_name,2000,lt=10)
most_p_old = get_most_p_old(file_name,2000,lt=10)
most_p_similar = get_similar_p_values(file_name,lt=10)
all_similar_names = most_p_similar['Name'].tolist()

most_n_young = get_most_n_young(file_name,2000,lt=10)
most_n_old = get_most_n_old(file_name,2000,lt=10)
sorted_old = sort_data(most_n_old,'old_p')
sorted_young = sort_data(most_n_young,'young_p')
all_old_names = sorted_old.head(49)['Name'].tolist()
all_young_names = sorted_young.head(429)['Name'].tolist()


new_young_female_black = ['Ciera', 'Nakia', 'Whitley', 'Mckenzie', 'Sierra', 'Shanice', 'Diamond', 'Domonique', 'Kourtney', 'Dominque', 'Asia', 'Shanika', 'Eboni', 'Ebony', 'Shayla', 'Shameka', 'Dominique', 'Dandre', 'Latoya', 'Courtney']
new_young_female_white = ['Kayleigh', 'Rhiannon', 'Fallon', 'Ashely', 'Kaley', 'Taylor', 'Jayla', 'Cassidy', 'Caitlin', 'Haley', 'Kayla', 'Carly', 'Ashley', 'Talia', 'Coty', 'Kacie', 'Jillian', 'Cheyenne', 'Codi', 'Emilee', 'Alyssa', 'Alysha', 'Lindsay', 'Jenna', 'Lauryn', 'Lacey', 'Jaclyn', 'Morgan', 'Marissa', 'Summer', 'Megan', 'Meghan', 'Maranda']

new_young_male_white = ['Connor', 'Tanner', 'Skylar', 'Keegan', 'Colton', 'Chase', 'Kyler', 'Conor', 'Tylor', 'Jakob', 'Caleb', 'Dylan', 'Austin', 'Tyler', 'Hunter', 'Cody', 'Jordon', 'Jairo', 'Zachary', 'Mohammad', 'Nikolas', 'Jacob', 'Hayden', 'Cole', 'Lucas', 'Colt', 'Dalton', 'Branden', 'Joshua']

new_young_male_black = ['Jaquan', 'Daquan', 'Jalen', 'Ladarius', 'Shaquille', 'Tre', 'Jaylen', 'Devante', 'Devonte', 'Rashad', 'Tevin', 'Deonte', 'Jamaal', 'Keon', 'Hakeem', 'Marquise', 'Trevon', 'Zechariah', 'Travon', 'Denzel', 'Javon', 'Dontae', 'Demario', 'Kameron', 'Demarcus', 'Isaiah', 'Deion', 'Davon', 'Tyrell', 'Dangelo', 'Bryson', 'Raheem', 'Dashawn', 'Malik', 'Darrius', 'Deandre']

new_old_female_black = ['Vickie', 'Laverne', 'Ethel','Wanda','Bessie']
new_old_female_white = ['Vicki', 'Debra', 'Tammy', 'Donna', 'Cathy', 'Peggy', 'Sheryl', 'Ronda', 'Lori', 'Carol', 'Sherri', 'Susan', 'Judy', 'Linda']

new_old_male_white = ['Tom', 'Dick', 'Ed', 'Tod', 'Rod']
new_old_male_black = ['Wendal','Alphonse','Jerome','Leroy','Terrence'] 


used_old_male_white = ['Tom', 'Dick', 'Ed', 'Tod', 'Rod']
used_old_male_black = ['Wendal','Alphonse','Jerome','Leroy','Terrence']
used_old_female_white = ['Vicki', 'Debra', 'Tammy', 'Donna', 'Cathy']
used_old_female_black = ['Vickie', 'Laverne', 'Ethel','Wanda','Bessie']
used_young_male_white = ['Connor', 'Tanner','Keegan', 'Colton', 'Kyler']
used_young_male_black = ['Jaquan', 'Daquan', 'Jalen', 'Ladarius', 'Shaquille']
used_young_female_white = ['Kayleigh', 'Rhiannon', 'Fallon', 'Ashely', 'Kaley']
used_young_female_black = ['Ciera', 'Nakia', 'Whitley','Sierra', 'Shanice']

for name in used_young_female_black+used_young_female_white+used_young_male_white+used_young_male_black:
    if name not in all_young_names:
        all_young_names.append(name)


for name in used_old_female_black+used_old_female_white+used_old_male_white+used_old_male_black:
    if name not in all_old_names:
        all_old_names.append(name)

