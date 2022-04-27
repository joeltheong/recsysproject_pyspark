import streamlit as st
import pandas as pd
import numpy as np
import SessionState
import os
from PIL import Image

import config
#from ingredient_parser import ingredient_parser

from word2vec_rec import KeywordRecommender

#import nltk

# Java
#import os
#os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk1.8.0_311"
#os.environ["SPARK_HOME"] = "C:/spark-3.2.1-bin-hadoop2.7"

# Initiating Spark Session
import findspark
findspark.init()
from pyspark.sql import SparkSession


# Initiating spark context
from pyspark import SparkConf
from pyspark import SparkContext
#sc = spark.sparkContext

# Evaluation metrics for recommender system 
#!pip install ml_metrics
#!pip install recmetrics

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt



# NLP
from operator import add
from pyspark.ml.feature import RegexTokenizer, CountVectorizer
from pyspark.ml.feature import StopWordsRemover, VectorAssembler
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.ml.feature import IDF
from pyspark.ml import Pipeline, PipelineModel

# SQL
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql.functions import split, col
from pyspark.sql import functions as f
from pyspark.sql import SparkSession

#def _initialize_spark() -> SparkSession:
    #conf = SparkConf().setAppName("recsysproject").config("spark.driver.memory", "2g").setMaster("local")
    #spark = SparkSession.builder.config(conf=conf).getOrCreate()
    #return spark

sc = SparkSession.builder.appName("word2vec").config("spark.driver.memory", "2g").getOrCreate()

indexed = sc.read.load("input/indexed.parquet")

def make_clickable(name, link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = name
    return f'<a target="_blank" href="{link}">{text}</a>'


def main():
    #image = Image.open("input/wordcloud.png").resize((680, 150))
    #st.image(image)
    st.markdown("# *Recipe Recommender System? :cooking:*")

    st.markdown(
        "An ML powered app by The Cloud Collective",
        unsafe_allow_html=True,
        #"An ML powered app by The Cloud Collective <a href='https://github.com/ongjoel/recsysproject' > <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/600px-Octicons-mark-github.svg.png' width='20' height='20' > </a> ",
        #unsafe_allow_html=True,
    )
    st.markdown(
        "## Feel like cooking something today based on the ingredients:tomato: you have at home:house:? "
    )
    st.markdown(
        "## Our ML-based model will scan through almost 100,000 recipes to recommend suitable ones... :mag: "
    )
    st.markdown(
        "## Try it below! :arrow_down:"
    )    
    st.text("")
    session_state = SessionState.get(
        recipe_df="",
        recipes="",
        model_computed=False,
        execute_recsys=False,
        recipe_df_clean="",
    )

    ingredients = st.text_input(
        "Enter ingredients you would like to cook with (seperated with a comma)",
        "butter, chicken, rice, prawns, carrots, garlic",
    )
    session_state.execute_recsys = st.button("Show me what you've got!")

    if session_state.execute_recsys:

        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            gif_runner = st.image("input/cooking_gif.gif")
        # recipe = rec_sys.RecSys(ingredients)
        recipe = KeywordRecommender(ingredients)
        recipe = recipe.toPandas()
        gif_runner.empty()
        #session_state.recipe_df_clean = recipe.select('*')
        session_state.recipe_df_clean = recipe.copy()        
        #session_state.recipe_df_clean = session_state.recipe_df_clean.toPandas()        
        # link is the column with hyperlinks
        #recipe["url"] = recipe.apply(
         #   lambda row: make_clickable(row["recipe"], row["url"]), axis=1
        #)
        #recipe_display = recipe[["rec_id", "score", "name", "ingredients", "steps"]]
        recipe_display = recipe[["name", "ingredients", "steps"]]        
        #recipe_display = recipe_display.toPandas()
        session_state.recipe_display = recipe_display.to_html(escape=False)
        session_state.recipes = recipe.name.values.tolist()
        #session_state.recipes = recipe.ingredients.values.tolist()        
        session_state.model_computed = True
        session_state.execute_recsys = False

    if session_state.model_computed:
        # st.write("Either pick a particular recipe or see the top 5 recommendations.")
        recipe_all_box = st.selectbox(
            "Either see the top 5 recommendations or pick a particular recipe ya fancy",
            ["Show me them all!", "Select a single recipe"],
        )
        if recipe_all_box == "Show me them all!":
            st.write(session_state.recipe_display, unsafe_allow_html=True)
        else:
            selection = st.selectbox(
                "Select a delicious recipe", options=session_state.recipes
            )
            #session_state.recipe_df_clean = session_state.recipe_df_clean.toPandas()
            selection_details = session_state.recipe_df_clean.loc[
                session_state.recipe_df_clean.name == selection
            ]
            #selection_details = session_state.recipe_df_clean.toPandas()
            #print(selection_details.shape)
            st.markdown(f"# {selection_details.name.values[0]}")
            #st.subheader(f"Website: {selection_details.url.values[0]}")
            #ingredients_disp = selection_details.ingredients.values[0].split(",")
            ingredients_disp = selection_details.ingredients.values[0]            
            
            st.subheader("Ingredients:")
            col1, col2 = st.columns(2)
            ingredients_disp = [
                ingred
                for ingred in ingredients_disp
                if ingred
                not in [
                    " skin off",
                    " bone out",
                    " from sustainable sources",
                    " minced",
                ]
            ]
            ingredients_disp1 = ingredients_disp[len(ingredients_disp) // 2 :]
            ingredients_disp2 = ingredients_disp[: len(ingredients_disp) // 2]
            for ingred in ingredients_disp1:
                col1.markdown(f"* {ingred}")
            for ingred in ingredients_disp2:
                col2.markdown(f"* {ingred}")
            # st.write(f"Score: {selection_details.score.values[0]}")

if __name__ == "__main__":
    main()