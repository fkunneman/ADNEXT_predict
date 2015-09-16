
import re
import json
import datetime

import docreader

month = {"Jan" : "01", "Feb" : "02", "Mar" : "03", "Apr" : "04", "May" : "05", "Jun" : "06", "Jul" : "07", 
    "Aug" : "08", "Sep" : "09", "Oct" : "10", "Nov" : "11", "Dec" : "12"}
date_time = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d+) (\d{2}:\d{2}:\d{2}) \+\d+ (\d{4})")

class Json_tweets_parser:

    def __init__(self, jsonfile):
        self.jsonfile = jsonfile
        self.lines = []
        self.columns = ['tweet_id', 'user_id', 'user_name', 'user_followers', 'user_location', 'date', 'time', 
            'reply_to_user', 'retweet_to_user', 'tweet_url', 'tweet_text']
        self.column_keys = [['id'], ['user', 'id'], ['user', 'screen_name'], ['user', 'followers_count'], 
            ['user', 'location'], ['created_at'], ['in_reply_to_screen_name'], ['retweeted_status', 'user', 
            'screen_name'], ['text']]
        
    def parse(self):
        dr = docreader.Docreader()
        self.lines = dr.parse_json(self.jsonfile, self.column_keys)

    def convert(self):
        newlines = []
        for line in self.lines:
            # set tweet_id and user_id to str
            line[0] = str(line[0])
            line[1] = str(line[1])
            # extract date and time object from 'created at'
            dt = date_time.search(line[5]).groups()
            timefields = [int(f) for f in dt[2].split(':')]
            date = datetime.date(int(dt[3]), int(month[dt[0]]), int(dt[1]))
            time = datetime.time(timefields[0], timefields[1], timefields[2])
            # generate tweet url
            url = 'https://twitter.com/' + line[2] + '/status/' + line[0]
            # assemble new line and add to new lines
            newlines.append(line[:5] + [date, time] + line[6:8] + [url, line[8]])
        self.lines = newlines
