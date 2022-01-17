# Reading-text
Recognize text in an image 

#### Assumptions: 
That these images have English words and sentences, each letter fits in a box that’s 16 pixels wide and 25 pixels tall. We’ll also assume that our documents only have the 26 uppercase latin characters, the 26 lowercase characters, the 10 digits, spaces, and 7 punctuation symbols, (),.-!?’".

The following two approaches were used for reading text from images:
1.	Naive Bayes method : For the emission probability, we used a simple naive Bayes classifier
2.	Viterbi algorithm

#### Firstly, we train the data, and store the emission, transition probabilities in a dictionary.

The dictionaries store:
1.	Probability of word given speech.
O1, ..., On and n hidden variables, l1..., ln, which are
the letters we want to recognize. We’re thus interested in P (l1, ..., ln|O1, ..., On).
2.	Transition probabilities 
The above probabilities are stored to use further for the algos.

#### How program works:
We have used the list of dictionaries to represent the Viterbi table emisson_pixel{}. The key to the dictionary component of letter we're working on right now. The Viterbi list's index denotes the letter for which we intend to locate the letter of test image. We calculate the maximum value of all parts of talks for a particular letter, and then assign that value to a cell in the Viterbi table (for given letter). The path that this algorithm takes is saved in another dictionary, and fresh parts of speech are added to the Viterbi path at each step. Finally, we compute the maximum and return the path for this maximum segment of speech, which is the path taken by our Viterbi method.
#### Challenges faced:

###### We comapring pixel values for emission probabilities between train image and test image. Due to high noise in images the white space was overpowering and hence to resolve this weighted sum was taken. Giving "*" more priority than " " (white space)
#### Design decisions: 

###### Data structure Dictionary – we used dictionaries because of the easy access to their values and constant time for fetching

#### Results of this evaluation on bc.train file and test image test_images/test-5-0.png were:

1. Simple: Opinion"of!the'Court
2. HMM: Opinion oflthe'Court
