import tweepy
import time
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Stream
from tweepy.streaming import StreamListener

class TwitterListener(StreamListener):

    def __init__(self, time_limit=60):
        
        self.start_time = time.time()
        self.limit = time_limit
        self.out_file = open('tweets.json', 'a', encoding='utf-8')
        super(TwitterListener, self).__init__()

    def on_data(self, data):
        if (time.time() - self.start_time) < self.limit:
            self.out_file.write(data)
            self.out_file.write("\n")
            return True
        #self.out_file.write('\n')
        else:
            print("Done")
            self.out_file.close()
            return False

    def on_error(self, status):
        print (status)

Consumer_Key = 'sRawaHIyoWzOv9GvvqJ86S0n6'
Consumer_Secret = 'yhyzTD3R2kya3ehBLxrbK9v2xgpTvz7lwL2BFVP6VXH4Z4HnIH'
Access_Token_Key = '918270035424002048-ssJLHLOpHj6a5BuBI5d3uEGOZFu56GB'
Access_Token_Secret = 'zJVd8YipoLgx255o6XQ8oIlPHlJ91MQE5SYTNroCMww32'


auth = tweepy.OAuthHandler(Consumer_Key,Consumer_Secret)
auth.set_access_token(Access_Token_Key, Access_Token_Secret)
#api = tweepy.API(auth)

##keyword = 'LA event'
##posts = api.search(q=keyword,count=5,languages=['en'])
##for tweet in posts:
##    j_tweet = tweet._json
##    for value in j_tweet.items():
##        #print(key)
##        print("* "+str(value))

keyword = ['los angeles']
twitterStream = Stream(auth, TwitterListener(time_limit=20)) 
twitterStream.filter(track=keyword, languages=['en'])
