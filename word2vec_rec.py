import sys
import logging
import unidecode
import ast

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

import config
#from ingredient_parser import ingredient_parser

# Java
#import os
#os.environ["JAVA_HOME"] = "C:/Program Files/Java/jdk1.8.0_311"
#os.environ["SPARK_HOME"] = "C:/spark-3.2.1-bin-hadoop2.7"

# Initiating Spark Session
import findspark
findspark.init()
from pyspark.sql import SparkSession
sc = SparkSession.builder.appName("word2vec").config("spark.driver.memory", "2g").getOrCreate()

# Initiating spark context
from pyspark import SparkConf
from pyspark import SparkContext
#sc = spark.sparkContext


# Evaluation metrics for recommender system 
#!pip install ml_metrics
#!pip install recmetrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import folium
import html

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

# Collaborative
#from pyspark.ml.recommendation import ALS, ALSModel
#from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, CrossValidator, CrossValidatorModel
#from pyspark.ml.evaluation import RegressionEvaluator

# Evaluation metrics
#import ml_metrics as metrics
#import recmetrics as met

#from sklearn.preprocessing import MinMaxScaler


indexed = sc.read.load("input/indexed.parquet")


def KeywordRecommender(key_words, sim_rec_limit=5):

  # load in data  
  #indexed = sc.read.load("F:/Visual Studio/recsysproject_pyspark/input/indexed.parquet")
  
  # load in word2vec model
  pipeline_mdl = PipelineModel.load("models/w2vmodel2" + 'pipe_txt')
  
  # Transform the recipes data
  recipes_pipeline_df = pipeline_mdl.transform(indexed)

  print('\nRecipes containing your ingredients: "' + key_words + '"')
    
  input_words_df = sc.sparkContext.parallelize([(0, key_words)]).toDF(['rec_id', 'ingredients'])
    
  # Transform the keywords to vectors
  input_words_df = input_words_df.withColumn("ingredients", f.regexp_replace('ingredients', "'|'", '')).withColumn("ingredients", split(col("ingredients"),", "))
  input_words_df = pipeline_mdl.transform(input_words_df)
    
  # Choose word2vec vectors
  input_key_words_vec = input_words_df.select('word_vec').collect()[0][0]

  # Function to calculate the cosine similarity of two vectors
  def CosineSim(vec1, vec2): 
      return np.dot(vec1, vec2) / np.sqrt(np.dot(vec1, vec1)) / np.sqrt(np.dot(vec2, vec2)) 

  # Get similarity
  recipe_vecs = recipes_pipeline_df.select('rec_id', 'word_vec').rdd.map(lambda x: (x[0], x[1])).collect()
  sim_rec_byword_rdd = sc.sparkContext.parallelize((i[0], float(CosineSim(input_key_words_vec, i[1]))) for i in recipe_vecs)

  sim_rec_byword_df = sc.createDataFrame(sim_rec_byword_rdd) \
         .withColumnRenamed('_1', 'rec_id') \
         .withColumnRenamed('_2', 'score') \
         .orderBy("score", ascending = False)
  
  # Return top 10 similar recipes
  rec_det = GetRecipeDetails(sim_rec_byword_df)
  rec_det.createOrReplaceTempView("tmp")
    
  # Filter out recommended recipes   
  query = '''SELECT * FROM tmp
  WHERE score >= 0.6
  '''
  
  filtered = sc.sql(query)
  df = filtered.orderBy("score", ascending = False).limit(sim_rec_limit)
  
  df = df.withColumn('rec_type', lit('Keyword'))

  return df


# A function to get the recipe details
def GetRecipeDetails(input_rec):
  
  a = input_rec.alias("a")
  b = indexed.alias("b")
  
  return a.join(b, col("a.rec_id") == col("b.rec_id"), 'inner') \
        .select([col('a.'+xx) for xx in a.columns] + [col('b.name'),col('b.ingredients'),
                                                      col('b.steps')])




