#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Himani Shah : shahhi@iu.edu
# (based on skeleton code by D. Crandall, Oct 2020)
#
# import Image 
#import Image
from PIL import Image, ImageDraw, ImageFont
import sys
import operator
CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

# Calculate emission probabilities:
# Using train image and test image (hidden and observed varaibles respectively)to get emission probabbilities.
def cal_emission_prob(train_letters,test_letters):
    emisson_prob = {}
    simple = []
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    a_list = list(range(len(test_letters)))

    # Generate dictionary of dictionary for each test image character and all corresponding train_letters
    for i in range(len(test_letters)):
        if a_list[i] not in emisson_prob.keys():
            emisson_prob[a_list[i]] = {}
    for i in range(len(test_letters)):
        for l in range(len(TRAIN_LETTERS)):
            if TRAIN_LETTERS[l] not in emisson_prob[i].keys():
                emisson_prob[i][TRAIN_LETTERS[l]] = 0
    
    # For each character in train_letters compare each pixel positions with each test-letters :
    # With each correct "*" and " " position add reward weight. 
    # More importance for "*" to handle highly noisy images  
    for i in range(len(test_letters)):
        for l in range(len(TRAIN_LETTERS)):         
            for j in range(CHARACTER_HEIGHT):
                for k in range(CHARACTER_WIDTH):                
                    if test_letters[i][j][k]== train_letters[TRAIN_LETTERS[l]][j][k] :
                        if test_letters[i][j][k] == "*":
                            emisson_prob[i][TRAIN_LETTERS[l]] += 0.85
                        else:
                            emisson_prob[i][TRAIN_LETTERS[l]] += 0.25

    # Return character with max probability                   
    for i in range(len(emisson_prob)):
        simple.append(max(emisson_prob[i].items(), key=operator.itemgetter(1))[0])
    return simple,emisson_prob

# Read data and preprocess : Here it is not done for files like bc.train with tagging included

def read_data(fname):
    
    file = open(fname, 'r');
    
    
    data = file.read()
   
    return data

# Calcualate the transitionn probabilities from givenn .txt file
def parse_train_txt(data):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    word_list = {}
    for i in range(len(data)):        
        if data[i] in word_list.keys():
            word_list[data[i]]['freq'] += 1
        else:
            word_list[data[i]] = {'freq': 1}
    for i in range(len(data)):     
        if i < len(data) -1:
            if data[i+1] in word_list[data[i]].keys():
                word_list[data[i]][data[i+1]] += 1
            else:
                word_list[data[i]][data[i+1]] = 1
    return word_list

#Implement HMM using emission and transition probabilities
def cal_Hmm(transition_probabilities,emisson_prob,emisson_probs,test_letters):
    hmm_pred = [] 
    hmm= {} # To store values for backtracking
    emisson_pixel ={} # To store values for each iteration 

    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    a_list = list(range(len(test_letters)))
    for i in range(len(test_letters)):
        if a_list[i] not in emisson_pixel.keys():
            emisson_pixel[a_list[i]] = {}
        if a_list[i] not in hmm.keys():
            hmm[a_list[i]] = {}
    for i in range(len(test_letters)):
        for l in range(len(TRAIN_LETTERS)):
            if TRAIN_LETTERS[l] not in emisson_pixel[i].keys():
                emisson_pixel[i][TRAIN_LETTERS[l]] = 0
                emisson_pixel[0][TRAIN_LETTERS[l]] = emisson_probs[0][TRAIN_LETTERS[l]]
            if TRAIN_LETTERS[l] not in hmm[i].keys():
                hmm[i][TRAIN_LETTERS[l]] = 0

    for i in range(1,len(test_letters)):
        for l in range(len(TRAIN_LETTERS)):
            maxi =[]           
            for j in range(len(TRAIN_LETTERS)):  
                if TRAIN_LETTERS[j] not in transition_probabilities.keys() or TRAIN_LETTERS[l] not in transition_probabilities[TRAIN_LETTERS[j]].keys():                  
                    maxi.append(emisson_probs[i][TRAIN_LETTERS[l]] * 0 * emisson_pixel[i-1][TRAIN_LETTERS[j]])
                else: 
                    maxi.append((emisson_probs[i][TRAIN_LETTERS[l]] *  emisson_pixel[i-1][TRAIN_LETTERS[j]]))
                    # maxi.append((transition_probabilities[TRAIN_LETTERS[j]][TRAIN_LETTERS[l]]/transition_probabilities[TRAIN_LETTERS[j]]['freq'] )* emisson_pixel[i-1][TRAIN_LETTERS[j]])
            x = maxi.index(max(maxi))
            # find maximum 
            emisson_pixel[i][TRAIN_LETTERS[l]] =  max(maxi)
            hmm[i][TRAIN_LETTERS[l]] = x  
    maxi =[]
    for i in range(len(TRAIN_LETTERS)):
        maxi.append(emisson_pixel[len(test_letters)-1][TRAIN_LETTERS[i]])
    x = maxi.index(max(maxi))
    hmm_pred.append(TRAIN_LETTERS[x]) 
    for i in range(len(test_letters)-1,0,-1):
       
        x = hmm[i][TRAIN_LETTERS[x]]
        hmm_pred.append(TRAIN_LETTERS[x])  
      
    hmm_pred1 = list(reversed(hmm_pred)) # Reverse the string
    return hmm_pred1





#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]

train_letters = load_training_letters(train_img_fname)

test_letters = load_letters(test_img_fname)

emisson_prob,emisson_probs = cal_emission_prob(train_letters,test_letters)
data = read_data(train_txt_fname)
transition_probabilities = parse_train_txt(data)
Hmm= cal_Hmm(transition_probabilities,emisson_prob,emisson_probs,test_letters)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print("\n".join([ r for r in train_letters['b'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# for i in range(len(test_letters)):
#     print("\n".join([ r for r in test_letters[i] ]))



# The final two lines of your output should look something like this:
print("Simple: "+ "".join(r for r in emisson_prob) )

print("   HMM: "+ "".join(r for r in Hmm))
# print("   HMM: " + "Sample simple result") 


