# LT2212 V19 Assignment 2

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: Hemanth Kumar Battula

## Additional instructions
The gendoc.py takes a folder name as input and prints .csv files as output based on the 
arguments used while running.
The simdoc.py takes .csv as input having column zero as index and all the columns represented
by feature names. The rows are vectors. Prints cosine similarities between different classes
to both console and output.txt file

## Results and discussion

###Vocabulary restriction.

Used vocabulary restriction to 20.
This makes the algorithm use the most common words which has more significance
in classifying the  documents than the less occurred which are insignificant.

### Result table


| File names                             | ('grain', 'grain')     | ('grain', 'crude')  | ('crude', 'grain')   | ('crude', 'crude')  |
|----------------------------------------|------------------------|---------------------|----------------------|---------------------|
| cv_outputfile_no_voc_restriction.csv   | 0.13262168793910395    | 0.0776268579319673  | 0.07762685793196729  | 0.16846136029580638 |
|----------------------------------------|------------------------|---------------------|----------------------|---------------------|
| cv_outputfile_restricted_voc_20.csv    | 0.37230896233716304	  | 0.25005894898638004 | 0.25005894898638     | 0.5232922357746077  |
|----------------------------------------|------------------------|---------------------|----------------------|---------------------|
| tfidf_outputfile_restricted_voc_20.csv | 0.3243782386313046	  | 0.16023570818171123 | 0.16023570818171123  | 0.4328152405665922  |
|----------------------------------------|------------------------|---------------------|----------------------|---------------------|
| tfidf_outputfile_no_voc_restriction.csv| 0.05478643002441667	  | 0.021621651047495197| 0.021621651047495197 | 0.056175959905801445|
|----------------------------------------|------------------------|---------------------|----------------------|---------------------|
| svdT_countvectoroutputfile_100.csv	 | 0.24763523060928347	  | 0.14283904427243252 | 0.14283904427243252  | 0.3040137260225901  |
|--------------------------------------- |------------------------|---------------------|----------------------|---------------------|
| svdT_countvectoroutputfile_1000.csv	 | 0.13318861923061648	  | 0.07776683914671656 | 0.07776683914671656  | 0.1688354782793359  |
|----------------------------------------|------------------------|---------------------|----------------------|---------------------|
| svdT_tfidfoutputfile_100.csv		 | 0.12985653358280963	  | 0.05929754450347478 | 0.05929754450347479  | 0.14895135157490838 |
|----------------------------------------|------------------------|---------------------|----------------------|---------------------|
| svdT_tfidfoutputfile_1000.csv	         | 0.0551517195649355	  | 0.021710958699288095| 0.02171095869928809  | 0.05646208224152123 |
|----------------------------------------|------------------------|---------------------|----------------------|---------------------|  
### The hypothesis in your own words
We try to classify documents belonging to two different classes based on cosine similarity.
As cosine similarity tends towards one, the similarity between two vectors is more. 
We first start by converting our documents to vectors. 
A vector representation is simply the numerical representation of words in the document. 
First we do the count of words vectors for each document. Then we find the cosine similarity 
using the count vectors of word embeddings of documents belonging to each class with itself and the other 
class as well. We also find the term frequency/Document inverse frequency of word embeddings of all
documents and calculate the cosine similarity. Finally to overcome the size of vectors,
we use the truncated SVD of the calculated vectors and calculate the cosine similarity. In the calculations 
done as I have eliminated the stop words, the cosine similarities seem very low as we have considered
single word vector and tf/idf counts. This is also known as one-hot encoding. There are almost 13 million
words in english vocabulary and we will be facing problems while classifying documents with large vectors.
Mostly with large vectors we will have zero's and rarely one's and two's in different positions representing
different words.  

### Discussion of trends in results in light of the hypothesis
When we choose the most common words, the cosine similarity is higher than compared
to the cosine similarities of document vectors using full vocabulary. Also when truncated SVD vectors are used
the one with lowest dimensions give more cosine similarity than the one with high dimensions.
Also, the count vectors cosine similarity analysis is much better compared to the tf/idf in this case.
This is because each word is considered individual and hence the similarity of words that produce
the same meaning would also yield different values.

## Bonus answers

The experiment is purely based on frequency based vectors.
In the experiment we calculated count vectors and tf/idf vectors. The tf/idf and count 
vectors provide word weights in document. But these frequency based vectors
 do not capture the word meaning in the document/corpus.
In the experiment we could have used n-grams to find the co reference word vectors than simply one words.
Stemming and lemmatization could have been applied to produce better results. For example in one-hot
encoding 'good' and 'better' are considered as two different vector values but lemmatization 
will allow it to consider as one making more better classifier. 
Considering co-occurrence with a fixed  context-window would also helped to achieve better results. This is because co-occurrence helps
words to preserve the meaningful relationship. These are called distributed embeddings.
These Embeddings makes word vectors to consider context. 
Prediction based vectors are more likely to produce better results than the frequency based vectors.
Prediction based vectors use probability to determine the relationship. Each word is given a probability to 
relate to a variable which is also a word. This done by creating weights based on co occurence vectors.
Two algorithms called Bag of words and skip-gram algorithm together are used to classify texts using
prediction based vectors.
