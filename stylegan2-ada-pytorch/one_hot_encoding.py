gender_list = ['male', 'female', 'both']
social_class_list = ['lower', 'middle', 'upper', 'all']
age_list = ['child', 'teen', 'young adult', 'adult', 'elder', 'all']
domain_list = ['agriculture', 'food and natural resources', 'architecture and construction', 'arts', 'audio and video technology',
               'communication', 'business and finance', 'education and training', 'government and public administration', 'medicine',
                 'software engineering', 'law', 'public safety and security', 'marketing', 'science', 'technology', 'engineering and math','animals','transportation','accessories and fashion','games and entertainment']
color_list = ['black','grey','blue','purple','red','brown','green','pink','orange','yellow']


gender = gender_list.index("male")
social_class = social_class_list.index("all")
age = age_list.index("elder")
domain = domain_list.index("agriculture")
color = color_list.index("green")

user_label = f"{gender},{social_class},{age},{domain},{color}"
