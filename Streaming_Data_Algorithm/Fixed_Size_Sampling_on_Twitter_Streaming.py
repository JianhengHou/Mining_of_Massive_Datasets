import tweepy
import random
import collections


class MyStreamListener(tweepy.StreamListener):
    def __init__(self):
        super(MyStreamListener, self).__init__()
        self.sequence_count = 0
        self.saved_list = []
        self.sumi = 0
        self.tag = collections.Counter()

    def on_status(self, status):
        tag_list = status.entities['hashtags']
        if len(tag_list) != 0:
            self.sequence_count += 1
            tag_list = [each["text"] for each in tag_list]
            # for first 100
            if self.sequence_count <= 100:
                self.saved_list.append(tag_list)
                for tag in tag_list:
                    self.tag[tag] += 1
            else:
                # after 100
                rmd = random.randint(1, self.sequence_count)
                if rmd <= 100:
                    random_index = random.randint(0, 99)
                    tags_to_move = self.saved_list[random_index]
                    for each in tags_to_move:
                        self.tag[each] -= 1
                    self.saved_list[random_index] = tag_list
                    for tag in tag_list:
                        self.tag[tag] += 1
            print("The number of tweets with tags from beginning", str(self.sequence_count))
            for tag, count in self.tag.most_common(3):
                print(tag + " : " + str(count))

consumer_key = 'kyHlpjEnzmOSPtRixkMxYktpy'
consumer_secret = 'uH0hLLGxvKbUZ65iuBF3Y6JcFtP58JzRtqTeoTm9ZoMBQdD9tB'
access_token = '1118037563237756929-J7YCiFXnxSKsXVr0XFsKAYNz5BxMca'
access_token_secret = 'g8hyuw3GGgpwZt70buxJJYg4e5PDPRC8XYFijPMRv5WxW'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)

tweets = myStream.filter(track="#")