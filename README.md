# Eluvio Challenge

------------------------------------------------------------------------------------------------------------

**Define the problem**

The problem I studied is to predict &#39;up\_votes&#39; based on &#39;date\_created&#39;, &#39;author&#39; and &#39;title&#39; columns.

One difficulty was, &#39;up\_votes&#39; ranges from 0 to 21253 and the distribution is pretty sparse for large up\_votes values. To deal with this outlier issue, I defined one classification problem and one regression problem, and trained them separately.

a) The binary classification problem

Define a binary class &#39;category&#39; based on: &#39;category&#39; 0 if &#39;up\_votes&#39; \&lt;= 5 else &#39;category&#39; 1. There is no specific reason of choosing threshold &#39;up\_votes&#39; = 5, but naively because in this way &#39;category&#39; = 0  and &#39;category&#39; = 1 have about equal data amounts. This makes it a balanced binary classification problem.

b) The regression problem

For regression problem, one has to deal with &#39;up\_votes&#39; outliers. The way I did is applying transformation &#39;logvotes&#39; = ln(1 + &#39;up\_votes&#39;) and use the &#39;logvotes&#39; as y\_label, where &#39;logvotes&#39; ranges from 0 to 9.96.

------------------------------------------------------------------------------------------------------------

**How to deal with large dataset**

For EDA and data processing purpose, one has to iterate over the whole &#39;.csv&#39; file by tools like Python generator. I used pd.read\_csv with chunksize = 10,000. To do DEA, I created some hashmaps to cache the information I need when iterating over 51 chunks.

For storing trainable np.array&#39;s and feeding them into nn model, I referred to the following blog:

[https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)

Let me state the brief idea. For each data, give it an ID. In our problem, the data in the i-th row has a natural ID i. When iterating over chunks, save the encoded trainable np.array of data i as a single file &#39;\_{i}\_.npy&#39; into the hard drive, also create a hashmap &#39;labels&#39; to cache {i: y\_label of data i}.

Create train\_list, val\_list, test\_list, storing ID&#39;s for each of training set, validation set and testing set. In our problem, there&#39;re in total 509,236 rows in the dataset. I splitted it into 400,000 training data, 100,000 validation data and 9,236 testing data, randomly and fixing random.seed(0).

Next, implement a DataGenerator class, build nn model and use &#39;fit\_generator&#39; in Keras to do training. DataGenerator can shuffle all data, load a batch\_size of data from saved &#39;.npy&#39; files and feed them into the model.

The benifits for doing so:

It always only accesses a batch\_size of data, so won&#39;t cause RAM overload;

Compare to accessing &#39;.csv&#39; file by pd.read\_csv with chunksize, saving each data in a single file allows shuffling during training.

Instead of saving each row of the original &#39;.csv&#39; as a single file, only saving encoded trainable np.array reduces hard drive memory cost.

------------------------------------------------------------------------------------------------------------

**Jobs of each &#39;.ipynb&#39; file**

a) &#39;EDA\_data\_processing.ipynb&#39;

It does EDA and saves the following files to &quot;D:\eluvio\data\_\&quot; for training use:

_&quot;modified\_embedding\_matrix.npy&quot;_ : word embedding matrix mapping word token to word embedding vector. I loaded pretrained &#39;glove.6B.50d&#39; for word embedding. There are 13741 missing words in &#39;title&#39; text. I tried SpellChecker but it works awful. Then I manually corrected the 40 most frequent words and set word embedding of the rest missing words be zero vectors.

_&quot;\_pad\_seq{i}.npy&quot; for i in range(_509236_)_ : size 50 np.array, padded word token for each title text.

_&quot;\_feature{i}.npy&quot; for i in range(_509236_)_ : size 8 np.array, features from &#39;date\_created&#39; and &#39;author&#39;: [&#39;author\_norm\_ave\_logvotes&#39;, &#39;author\_norm\_pub\_count&#39;, &#39;norm\_year&#39;, &#39;month\_cosine&#39;, &#39;month\_sine&#39;, &#39;weekday\_cosine&#39;, &#39;weekday\_sine&#39;, &#39;author\_ave\_category&#39;]. To avoid cheating, the author-related features are calculated only for authors in training dataset. For those authors appear in validation or testing dataset but not in training dataset, author-related features are set to be zero.

_&quot;logvotes\_labels.npy&quot;, &quot;category\_labels.npy&quot;_ : y\_labels

b) &#39;LSTM\_GloVe\_feature\_regression&#39;

Regression model predicting logvotes.

c) &#39;LSTM\_GloVe\_feature\_classification&#39;

Classification model predicting category.

------------------------------------------------------------------------------------------------------------

**Output**

Please refer to &#39;LSTM\_GloVe\_feature\_regression&#39; and &#39;LSTM\_GloVe\_feature\_classification&#39;.

Those two model files should have explained themselves well.
