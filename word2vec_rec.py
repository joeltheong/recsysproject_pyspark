import sys
import logging
import ast
from collections import defaultdict
import config

# Initiating Spark Session
import findspark
findspark.init()
from pyspark.sql import SparkSession
sc = SparkSession.builder.appName("word2vec").config("spark.driver.memory", "2g").getOrCreate()
logging.getLogger("py4j").setLevel(logging.ERROR)

# Initiating spark context
from pyspark import SparkConf
from pyspark import SparkContext

import numpy as np
import pandas as pd
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

indexed = sc.read.parquet("input/indexed.parquet")
#indexed.createOrReplaceTempView("indexed")
#indexed.persist()
#@st.cache(allow_output_mutation=True, suppress_st_warning=True, hash_funcs={"MyUnhashableClass": lambda _: None})

#indexed = load_indexed()
pipeline_mdl = PipelineModel.load("models/w2vmodel2" + 'pipe_txt')

def KeywordRecommender(key_words, sim_rec_limit=5):
  
  # load in word2vec model
  #pipeline_mdl = PipelineModel.load("models/w2vmodel2" + 'pipe_txt')
  #pipeline_mdl.createOrReplaceTempView("pipeline_mdl")
  #pipeline_mdl.persist()
  
  # Transform the recipes data
  recipes_pipeline_df = pipeline_mdl.transform(indexed).cache()
  #recipes_pipeline_df.createOrReplaceTempView("recipes_pipeline_df")
  #recipes_pipeline_df.persist()

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




