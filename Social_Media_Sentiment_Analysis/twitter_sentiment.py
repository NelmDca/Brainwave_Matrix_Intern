from transformers import AutoTokenizer,AutoModelForSequenceClassification
from scipy.special import softmax
import emoji
happy_face_emoji = "😊"

tweet = "@NelumDissanayake Great Content!" + happy_face_emoji + "https://google.com"


tweet_words = []

for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'

    elif word.startswith('http'):
        word = "http"
    tweet_words.append(word)

twwt_proc = " ".join(tweet_words)
#print(twwt_proc)

#load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

#sentiment analysis
encoded_tweet = tokenizer(twwt_proc, return_tensors='pt')
#print(encoded_tweet)

output= model(encoded_tweet['input_ids'],encoded_tweet['attention_mask'])
#print(output)

scores = output[0][0].detach().numpy()
scores = softmax(scores)
#print(scores)

for i in range(len(scores)):
    l = labels[i]
    s = scores[i]

    print(l,s)