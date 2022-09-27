#!/usr/bin/env python
# coding: utf-8

# # DS/CMPSC 410 Fall 2022
# # Instructor: Professor John Yen
# # TA: Rupesh Prajapati and Haiwen Guan
# # LA: Zining Yin
# # Lab 5: Data Frames, SQL Functions, DF-based Join, and Top Movie Reviews 
# 
# # The goals of this lab are for you to be able to
# ## - Use Data Frames in Spark for Processing Structured Data
# ## - Perform Basic DataFrame Transformation: Filtering Rows and Selecting Columns of DataFrame
# ## - Create New Column of DataFrame using `withColumn`
# ## - Use DF SQL Function split to transform a string into an Array
# ## - Filter on a DF Column that is an Array using `array_contains`
# ## - Perform Join on DataFrames 
# ## - Use GroupBy, followed by count and sum DF transformation to calculate the count and the sum of a DF column (e.g., reviews) for each group (e.g., movie).
# ## - Perform sorting on a DataFrame column
# ## - Apply the obove to find Movies in a Genre that has good reviews with a significant number of ratings (use 10 as the threshold for local mode, 100 as the threshold for cluster mode).
# ## - After completing all exercises in the Notebook, convert the code for processing large reviews dataset and large movies dataset to find Drama movies with top average ranking with at least 100 reviews.
# 
# ## Total Number of Exercises: 
# - Exercise 1: 5 points
# - Exercise 2: 5 points
# - Exercise 3: 5 points
# - Exercise 4: 10 points
# - Exercise 5: 10 points
# - Exercise 6: 5 points
# - Exercise 7: 10 points
# - Exercise 8: 10 points
# - Exercise 9: 10 points
# - Part B: 30 points (complete spark-submit in the cluster)
# ## Total Points: 100 points
# 
# # Due: midnight, September 25, 2022

# ## The first thing we need to do in each Jupyter Notebook running pyspark is to import pyspark first.

# In[154]:


import pyspark


# ### Once we import pyspark, we need to import "SparkContext".  Every spark program needs a SparkContext object
# ### In order to use Spark SQL on DataFrames, we also need to import SparkSession from PySpark.SQL

# In[301]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row


# ## We then create a Spark Session variable (rather than Spark Context) in order to use DataFrame. 
# - Note: We temporarily use "local" as the parameter for master in this notebook so that we can test it in ICDS Roar.  However, we need to change "local" to "Yarn" before we submit it to XSEDE to run in cluster mode.

# In[302]:


ss=SparkSession.builder.appName("Lab 5 Top Reviews").getOrCreate()


# In[303]:


ss.sparkContext.setCheckpointDir("~/scratch")


# # Exercise 1 (5 points) 
# - (a) Add your name below AND 
# - (b) replace the path below in both `ss.read.csv` statements with the path of your home directory.
# 
# ## Answer for Exercise 1 (Double click this Markdown cell to fill your name below.)
# - a: Student Name: Michael Jabco
# 

# In[304]:


rating_schema = StructType([ StructField("UserID", IntegerType(), False ),                             StructField("MovieID", IntegerType(), True),                             StructField("Rating", FloatType(), True ),                             StructField("RatingID", IntegerType(), True ),                            ])


# In[305]:


ratings_DF = ss.read.csv("/storage/home/mvj5144/Lab5/ratings_2.csv", schema= rating_schema, header=True, inferSchema=False)
# In the cluster mode, we need to change to  `header=False` because it does not have header.


# In[306]:


movie_schema = StructType([ StructField("MovieID", IntegerType(), False),                             StructField("MovieTitle", StringType(), True ),                             StructField("Genres", StringType(), True ),                            ])


# In[307]:


movies_DF = ss.read.csv("/storage/home/mvj5144/Lab5/movies_2.csv", schema=movie_schema, header=True, inferSchema=False)
# In the cluster mode, we need to change to `header=False` because it does not have header.


# In[308]:


movies_DF.printSchema()


# In[309]:


movies_DF.show(10)


# In[310]:


ratings_DF.printSchema()


# In[311]:


ratings_DF.show(5)


# # 2. DataFrames Transformations
# DataFrame in Spark provides higher-level transformations that are convenient for selecting rows, columns, and for creating new columns.  These transformations are part of Spark SQL.
# 
# ## 2.1 `where` DF Transformation for Filtering/Selecting Rows
# Select rows from a DataFrame (DF) that satisfy a condition.  This is similar to "WHERE" clause in SQL query language.
# - One important difference (compared to SQL) is we need to add `col( ...)` when referring to a column name. 
# - The condition inside `where` transformation can be an equality test, `>` test, or '<' test, as illustrated below.

# # `show` DF action
# The `show` DF action is similar to `take` RDD action. It takes a number as a parameter, which is the number of elements to be randomly selected from the DF to be displayed.

# In[312]:


movies_DF.where(col("MovieTitle")== "Toy Story (1995)").show()


# In[313]:


ratings_DF.where(col("Rating") > 3).show(5)


# # `count` DF action
# The `count` action returns the total number of elements in the input DataFrame.

# In[314]:


ratings_DF.filter(4 < col("Rating")).count()


# # Exercise 2 (5 points) Filtering DF Rows
# ### Complete the following statement to (1) select the `ratings_DF` DataFrame for reviews that are exactly 5, and (2) count the total number of such reviews.

# In[315]:


review_5_count = ratings_DF.where( col("Rating") == 5).count()
print(review_5_count)


# ## 2.2 DataFrame Transformation for Selecting Columns
# 
# DataFrame transformation `select` is similar to the projection operation in SQL: it returns a DataFrame that contains all of the columns selected.

# In[316]:


movies_DF.select("MovieTitle").show(5)


# In[317]:


movies_DF.select(col("MovieTitle")).show(5)


# # Exercise 3 (5 points) Selecting DF Columns
# ## Complete the following PySpark statement to (1) select only `MovieID` and `Rating` columns, and (2) save it in a DataFrame called `movie_rating_DF`.

# In[318]:


movie_rating_DF = ratings_DF.select("MovieID","Rating")


# In[319]:


movie_rating_DF.show(5)


# # 2.3 Statistical Summary of Numerical Columns
# DataFrame provides a `describe` method that provides a summary of basic statistical information (e.g., count, mean, standard deviation, min, max) for numerical columns.

# In[320]:


ratings_DF.describe().show()


# ## RDD has a histogram method to compute the total number of rows in each "bucket".
# The code below selects the Rating column from `ratings_DF`, converts it to an RDD, which maps to extract the rating value for each row, which is used to compute the total number of reviews in 5 buckets.

# In[321]:


ratings_DF.select(col("Rating")).rdd.map(lambda row: row[0]).histogram([0,1,2,3,4,5,6])


# # 3. Transforming the Generes Column into Array of Generes 
# ## We want transform a column Generes, which represent all Generes of a movie using a string that uses "|" to connect the Generes so that we can later filter for movies of a Genere more efficiently.
# ## This transformation can be done using `split` Spark SQL function (which is different from python `split` function)

# In[322]:


Splitted_Generes_DF= movies_DF.select(split(col("Genres"), '\|'))
Splitted_Generes_DF.show(5)


# ## 3.1 Adding a Column to a DataFrame using withColumn
# 
# # `withColumn` DF Transformation
# 
# We often need to transform content of a column into another column. For example, it is desirable to transform the column Genres in the movies DataFrame into an `Array` of genres that each movie belongs, we can do this using the DataFrame method `withColumn`.

# # Exercise 4 (10 points)
# Complete the code below to create a new column called "Genres_Array", whose values are arrays of genres for each movie.

# In[323]:


moviesG_DF= movies_DF.withColumn("Genres_Array",split(col("Genres"), "\|"))


# In[324]:


moviesG_DF.printSchema()


# In[325]:


moviesG_DF.show(5)


# # An DF-based approach to compute Average Movie Ratings and Total Count of Reviews for each movie.

# # `groupBy` DF transformation
# Takes a column name (string) as the parameter, the transformation groups rows of the DF based on the column.  All rows with the same value for the column is grouped together.  The result of groupBy transformation is often folled by an aggregation across all rows in the same group.  
# 
# # `sum` DF transformation
# Takes a column name (string) as the parameter. This is typically used after `groupBy` DF transformation, `sum` adds the content of the input column of all rows in the same group.
# 
# # `count` DF transformation
# Returns the number of rows in the DataFrame.  When `count` is used after `groupBy`, it returns a DataFrame with a column called "count" that contains the total number of rows for each group generated by the `groupBy`.

# In[326]:


Movie_RatingSum_DF = ratings_DF.groupBy("MovieID").sum("Rating")


# In[327]:


Movie_RatingSum_DF.show(4)


# # Exercise 5 (10 points)
# Complete the code below to calculate the total number of reviews for each movies.

# In[328]:


Movie_RatingCount_DF = ratings_DF.groupBy("MovieID").count()


# In[329]:


Movie_RatingCount_DF.show(4)


# # Exercise 6 (5 points)
# Complete the code below to perform DF-based inner join on the column MovieID.

# In[330]:


Movie_Rating_Sum_Count_DF = Movie_RatingSum_DF.join(Movie_RatingCount_DF,'MovieID','inner')


# In[331]:


Movie_Rating_Sum_Count_DF.show(4)


# In[332]:


Movie_Rating_Count_Avg_DF = Movie_Rating_Sum_Count_DF.withColumn("AvgRating", (col("sum(Rating)")/col("count")))


# In[333]:


Movie_Rating_Count_Avg_DF.show(4)


# # 5. Join Transformation on Two DataFrames
# We want to join the avg_rating_total_review_DF with moviesG_DF

# In[334]:


joined_DF = Movie_Rating_Count_Avg_DF.join(moviesG_DF,'MovieID', 'inner')


# In[335]:


moviesG_DF.printSchema()


# In[336]:


joined_DF.printSchema()


# In[337]:


joined_DF.show(4)


# # 6. Filter DataFrame on an Array Column of DataFrame Using `array_contains`
# 
# ## Exercise 7 (10 points)
# Complete the following code to filter for Animation movies.

# In[338]:


from pyspark.sql.functions import array_contains
Drama_DF = joined_DF.filter(array_contains('Genres_Array',                                                "Drama")).select("MovieID","AvgRating","count","MovieTitle")


# In[339]:


Drama_DF.show(5)


# In[340]:


Drama_DF.count()


# In[341]:


Drama_DF.describe().show()


# In[342]:


Sorted_Drama_DF = Drama_DF.orderBy('AvgRating', ascending=False)


# In[343]:


Sorted_Drama_DF.show(10)


# # Exercise 8 (10 points)
# Use DataFrame method `where` or `filter` to find all drama movies that have more than 10 reviews.

# In[344]:


Top_Drama_DF = Sorted_Drama_DF.where(col('count') > 10)


# In[ ]:





# In[345]:


Top_Drama_DF.show(5)


# ## Exercise 9 (10 ponts)
# Complete the code below to save the Drama Movies, ordered by average rating, that received more than 100 reviews.

# In[346]:



output_path = "/storage/home/mvj5144/Lab5/Lab5_Sorted_Top_Drama_Movies_local2"
Top_Drama_DF.write.csv(output_path)


# In[347]:


Top_Drama_DF = Sorted_Drama_DF.where(col('count') > 100)


# In[348]:


Top_Drama_DF.show(5)


# In[349]:


#ss.stop()


# # Appendix A
# ## An RDD-based Approach for Computing Total Reviews and Average Rating for Each Movie
# Because it is convenient and efficient to compute both total reviews and average rating for each movie using key value pairs, we will convert the reviews Data Frame into RDD.

# In[350]:



ratings_RDD = ratings_DF.rdd
#ratings_RDD.take(3)


# In[351]:


movie_ratings_RDD = ratings_RDD.map(lambda row: (row.MovieID, row.Rating))


# In[352]:


#movie_ratings_RDD.take(4)


# In[353]:


movie_review_1_RDD = movie_ratings_RDD.map(lambda x: (x[0], 1))
movie_review_total_RDD = movie_review_1_RDD.reduceByKey(lambda x, y: x+y, 1)


# In[354]:


#movie_review_total_RDD.take(4)


# In[355]:


# Compute average rating for each movie
rating_total_RDD = movie_ratings_RDD.reduceByKey(lambda x, y: x+y, 1)


# In[356]:


#rating_total_RDD.take(4)


# ## Join Transformation on Two RDDs
# Two Key Value Pairs RDDs can be joined on the RDD (similar to the join operation in SQL) to return a new RDD, whose rows is an inner join of the two input RDDs.  Only key value pairs occur in both input RDDs occur in the output RDD.

# In[357]:


# Join the two RDDs (one counts total reviews, the other computes sum of ratings)
joined_RDD = rating_total_RDD.join(movie_review_total_RDD)


# In[358]:


#joined_RDD.take(4)


# # The following code computes average rating for each movie from the joined RDD.

# In[359]:


# Compute average rating for each movie
average_rating_RDD = joined_RDD.map(lambda x: (x[0], x[1][0]/x[1][1] ))


# In[360]:


#average_rating_RDD.take(4)


# ### The following code joins the average_rating_RDD with movie_review_total_RDD so that we obtain an RDD in the form of 
# ```
# (<movieID>, (<average rating>, <total review>)
# ```
# ### because we want to keep both average rating and total review of each movie so that we can filter on either of them.

# In[361]:


# We want to keep both average review and total number of reviews for each movie. 
# So we do another join her.
avg_rating_total_review_RDD = average_rating_RDD.join(movie_review_total_RDD)


# In[362]:


#avg_rating_total_review_RDD.take(4)


# ## Transforming RDD to Data Frame
# An RDD can be transformed to a Data Frame using toDF().  We want to transform the RDD containing average rating and total reviews for each movie into a Data Frame so that we can answer questions that involve both movie reviews and generes such as the following:
# - What movies in a genre (e.g., comedy) has a top 10 average review among those that receive at least k reviews?

# In[363]:


# Before transforming to Data Frame, we first convert the key value pairs of avg_rating_total_reivew_RDD 
# which has the format of (<movie ID> (<average rating> <review total>) )  to a tuple of the format
# (<movie ID> <average rating> <review total>)
avg_rating_total_review_tuple_RDD = avg_rating_total_review_RDD.map(lambda x: (x[0], x[1][0], x[1][1]) )


# In[364]:


#avg_rating_total_review_tuple_RDD.take(4)


# ## Defining a Schema for Data Frame
# As we have seen before, each Data Frame has a Schema, which defines the names of the column and the type of values for the column (e.g., string, integer, or float).  There are two ways to specify the schema of a Data Frame:
# - Infer the schema from the heading and the value of an input file (e.g., CSV).  This is how the schema of movies_DF was created in the beginning of this notebook.
# - Explicitly specify the Schema
# We will use one approach in the second category here to specify the column names and the type of column values of the DataFrame to be converted from the RDD above.

# # Exercise 7 (10 points)
# Define a schema and use it to convert the `avg_rating_total_reive_tuple_RDD` to a DataFrame.

# In[365]:


schema = StructType([ StructField("MovieID", IntegerType(), True ),                      StructField("AvgRating", FloatType(), True ),                      StructField("TotalReviews", IntegerType(), True)                     ])


# In[366]:


# Convert the RDD to a Data Frame
avg_review_DF = avg_rating_total_review_tuple_RDD.toDF(schema)


# In[367]:


avg_review_DF.printSchema()


# In[368]:


#avg_review_DF.take(4)


# In[369]:



columns = ['MovieID','AvgRating','TotalReviews']
dataframe = ss.createDataFrame(avg_rating_total_review_tuple_RDD,columns)


# In[370]:


dataframe.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




