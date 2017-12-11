import tweepy
import time
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Stream
from tweepy.streaming import StreamListener
import sys
import ast

Consumer_Key = 'vGk3JRVlO0YwnzjVBm69fGqJX'
Consumer_Secret = '9T97moIojHnN3rDAcFyv03suJoP4T3KEo15tlYMWK3x0UMRfBl'
Access_Token_Key = '925772963206504448-rBdE0DaVbQ0Ilrij9dGc52w3JWqcGs2'
Access_Token_Secret = 'GEWaDK7TPPDj4ufLDswrajfERduJlHDClowkaQBmlIwYL'

class TwitterListener(StreamListener):

    def __init__(self, time_limit=60):
        
        self.start_time = time.time()
        self.limit = time_limit
        self.out_file = open('tweets.json', 'a', encoding='utf-8')
        self.processed_data = []
        super(TwitterListener, self).__init__()

    def on_data(self, data):
        if (time.time() - self.start_time) < self.limit:
            self.out_file.write(data)
            process_json(data, self.processed_data)
            self.out_file.write('\n')
            return True
        #self.out_file.write('\n')
        else:
            print('Done')
            print(self.processed_data)
            self.out_file.close()
            return False

    def on_error(self, status):
        print (status)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '-r':
        reprocess_json()
    else:
        
        auth = tweepy.OAuthHandler(Consumer_Key,Consumer_Secret)
        auth.set_access_token(Access_Token_Key, Access_Token_Secret)    
        api = tweepy.API(auth)
        keyword = '"los angeles" OR la disaster OR earthquake OR "forest fire" OR flood OR hurricane OR tsunami'
        #keyword = 'los angeles Dodgers OR Lakers OR Clippers OR Ram'
        #keyword = "los angeles traffic OR car accident"
        #keyword = "los angeles traffic OR festival"
        out_raw_file_name = '../data/out/raw_tweets.json'
        results = api.search(q=keyword, count=100, languages=['en'])
        out_file_raw = open(out_raw_file_name, 'a')
        for r in results:
            out_file_raw.write(str(r._json))
            out_file_raw.write('\n')
        out_file_raw.close()
    
#twitterStream = Stream(auth, TwitterListener(time_limit=20)) 
#twitterStream.filter(track=keyword, languages=['en'])

