

import pandas as pd
import nltk
import string
import ast
import re
import unidecode

# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import config

# Weigths and measures are words that will not add value to the model. I got these standard words from
# https://en.wikibooks.org/wiki/Cookbook:Units_of_measurement

# # We lemmatize the words to reduce them to their smallest form (lemmas).
# lemmatizer = WordNetLemmatizer()
# measures = [lemmatizer.lemmatize(m) for m in measures]
# words_to_remove = [lemmatizer.lemmatize(m) for m in words_to_remove]


def ingredient_parser(ingreds):
    # measures and common words (already lemmatized)   
    #measures = ['']
    #words_to_remove = ['']
    # Turn ingredient list from string into a list 
    if isinstance(ingreds, list):
        ingredients = ingreds
    else:
        ingredients = ast.literal_eval(ingreds)
    # We first get rid of all the punctuation
    translator = str.maketrans("", "", string.punctuation)
    # initialize nltk's lemmatizer    
    lemmatizer = WordNetLemmatizer()
    ingred_list = []
    for i in ingredients:
        i.translate(translator)
        # We split up with hyphens as well as spaces
        items = re.split(" |-", i)
        # Get rid of words containing non alphabet letters
        items = [word for word in items if word.isalpha()]
        # Turn everything to lowercase
        items = [word.lower() for word in items]
        # remove accents
        #items = [unidecode.unidecode(word) for word in items]
        # Lemmatize words so we can compare words to measuring words
        items = [lemmatizer.lemmatize(word) for word in items]
        # get rid of stop words
        #stop_words = set(corpus.stopwords.words('english'))
        #items = [word for word in items if word not in stop_words]
        # Gets rid of measuring words/phrases, e.g. heaped teaspoon
        #items = [word for word in items if word not in measures]
        # Get rid of common easy words
        #items = [word for word in items if word not in words_to_remove]
        if items:
            ingred_list.append(" ".join(items))
    #ingred_list = " ".join(ingred_list)            
    return ingred_list


if __name__ == "__main__":
    recipe_df = pd.read_csv(config.RECIPES_PATH)
    recipe_df["parsed"] = recipe_df["ingredients"].apply(
        lambda x: ingredient_parser(x))
    df = recipe_df[["name", "steps", "parsed", "ingredients"]]
    #df = recipe_df.dropna()

    # remove - Allrecipes.com from end of every recipe title
    #m = df.recipe_name.str.endswith("Recipe - Allrecipes.com")
    #df["recipe_name"].loc[m] = df.recipe_name.loc[m].str[:-23]
    df.to_csv(config.PARSED_PATH, index=False)

    # vocabulary = nltk.FreqDist()
    # for ingredients in recipe_df['ingredients']:
    #     ingredients = ingredients.split()
    #     vocabulary.update(ingredients)

    # for word, frequency in vocabulary.most_common(200):
    #     print(f'{word};{frequency}')
    # fdist = nltk.FreqDist(ingredients)

    # common_words = []
    # for word, _ in vocabulary.most_common(250):
    #     common_words.append(word)
    # print(common_words)
