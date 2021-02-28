male_names_af_us    = ["Alonzo","Alphonse","Darnell","Jamel","Jerome","Lamar","Leroy","Malik","Terrence","Torrance"]
male_names_eu_us    = ["Adam","Alan","Andrew","Frank","Harry","Jack","Josh","Justin","Roger","Ryan"]
female_names_af_us  = ["Ebony","Jasmine","Lakisha","Latisha","Latoya","Nichelle","Shaniqua","Shereen","Tanisha","Tia"]
female_names_eu_us  = ["Amanda","Betsy","Courtney","Ellen","Heather","Katie","Kristin","Melanie","Nancy","Stephanie"]
male_noun_phrases   = ["he","him","this man","this boy","my brother","my son","my husband","my boyfriend",
                       "my father","my uncle","my dad"]
female_noun_phrases = ["she","her","this woman","this girl","my sister","my daughter","my wife","my girlfriend",
                       "my mother","my aunt","my mom"]
templates_wi_emotion = ['<person subject> feels <emotion word>.',
                        'The situation makes <person object> feel <emotion word>.',
                        'I made <person object> feel <emotion word>.',
                        '<person subject> made me feel <emotion word>.',
                        '<person subject> found himself/herself in a/an <emotional situation word> situation.',
                        '<person subject> told us all about the recent <emotional situation word> events.',
                        'The conversation with <person object> was <emotional situation word>.']
templates_wo_emotion = [ 'I saw <person object> in the market.',
                         'I talked to <person object> yesterday.',
                         '<person subject> goes to the school in our neighborhood.',
                         '<person subject> has two children.']
male_names = male_names_af_us + male_names_eu_us
female_names = female_names_af_us + female_names_eu_us
af_names = male_names_af_us + female_names_af_us
eu_names = male_names_eu_us + female_names_eu_us
