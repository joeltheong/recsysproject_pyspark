import streamlit as st
import pandas as pd
import numpy as np
import SessionState
import os
from PIL import Image
import config
from word2vec_rec import KeywordRecommender

# Initiating Spark Session
import findspark
findspark.init()
from pyspark.sql import SparkSession

# Initiating spark context
from pyspark import SparkConf
from pyspark import SparkContext

import numpy as np
import pandas as pd

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

sc = SparkSession.builder.appName("word2vec").config("spark.driver.memory", "2g").getOrCreate()

indexed = sc.read.load("input/indexed.parquet")

def make_clickable(name, link):
    text = name
    return f'<a target="_blank" href="{link}">{text}</a>'


def main():
    st.markdown("# *Recipe Recommender System?:knife_fork_plate:*")

    st.markdown(
        "An ML powered app by The Cloud Collective:cloud:",
        unsafe_allow_html=True,
    )
    st.markdown(
        "## Feel like cooking something today based on the ingredients:carrot: you have at home:house_with_garden:?"
    )
    st.markdown(
        "## Our ML-based model will scan through thousands of recipes to recommend suitable ones... :eyes:"
    )
    st.markdown(
        "## Try it below! :point_down:"
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
        recipe = KeywordRecommender(ingredients)
        recipe = recipe.toPandas()
        gif_runner.empty()
        session_state.recipe_df_clean = recipe.copy()        
        recipe_display = recipe[["name", "ingredients", "score"]]        
        session_state.recipe_display = recipe_display.to_html(escape=False)
        session_state.recipes = recipe.name.values.tolist()   
        session_state.model_computed = True
        session_state.execute_recsys = False

    if session_state.model_computed:

        recipe_all_box = st.selectbox(
            "See all recommendations or select a single recipe for more details",
            ["Summary", "Select recipe"],
        )
        if recipe_all_box == "Summary":
            st.write(session_state.recipe_display, unsafe_allow_html=True)
        else:
            selection = st.selectbox(
                "Select recipe", options=session_state.recipes
            )

            selection_details = session_state.recipe_df_clean.loc[
                session_state.recipe_df_clean.name == selection
            ]

            st.markdown(f"# {selection_details.name.values[0]}")
            ingredients_disp = selection_details.ingredients.values[0]
            steps_disp = selection_details.steps.values[0]                            
            
            st.subheader("Score:100::")            
            st.write(f"{selection_details.score.values[0]}")

            st.subheader("Ingredients:pineapple::")
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

            st.subheader("Steps::memo:")
            st.markdown(steps_disp)

if __name__ == "__main__":
    main()