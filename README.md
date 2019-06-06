# MSCi 641: Text Analytics Assignment 2

## Result:

Best Alpha for Without Stopwords: 0.8  
Best Alpha for With Stopwords: 0.4  

--------------------------------------------------  
With Stopwords:  
--------------------------------------------------  
Unigrams: 0.8068125  
Bigrams: 0.82605  
Unigrams+Bigrams: 0.8336375  
--------------------------------------------------  
Without Stopwords:  
--------------------------------------------------  
Unigrams: 0.8041  
Bigrams: 0.791375  
Unigrams+Bigrams: 0.82225  


## Analysis

1) The MNB algorithm performed better with stopwords. After inspecting the reviews in the Amazon dataset, we can clearly see that stopwords like couldn't, shouldn't, hasn't, don't, again, against, more, most, but, etc. convey the users' sentiment for the review. By removing these stopwords, we would be stripping that extra information we would have gotten. In terms of sentiment analysis, the stopwords contribute either directly towards the sentiment by supporting or inflecting the previous words, or by providing valuable context to the non stopword-words.   

2) In both the cases, the MNB algorithm performed better with a combination of Unigram and Bigram word counts. We know that lower "n"s in "n-gram" correspond to capturing syntactic representation of word sequences while higher "n"s correspond to capturing semantic representation of word sequences. Unigrams, being the lowest syntactic and semantic representation, which justifies why it gives the lowest accuracy. Using bigrams increases the quality of semantic association with the
sentiment of the review, hence its accuracy is slightly better than unigrams. Using bigrams though would likely decrease the frequency counts as compared to unigrams since many bigrams might not be seen in multiple reviews, unlike unigrams. Hence combining both unigram and bigram features would contribute better for determining the sentiment of the review. It would be interesting to experiment with interpolation here and actually see the optimal contribution fractions.
