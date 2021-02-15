import numpy as np
import tensorflow as tf

# These should be the only tensorflow classes you need:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

# get_data returns (train_x, train_y), (test_x, test_y)
# argument determines whether images are shifted to top-left or bottom-right
# X values are an array of 30x30 images
# Y values are an array of 10 one-hot encoded labels
from bs4 import BeautifulSoup
from requests import get
from urllib.request import urlopen
import numpy
import csv
import time

lyricsvector = [] #input (bag of words)
genrevector = [] #target
songinfovector = []  #metadata (artist and songname)

urllist = [
"http://www.songlyrics.com/news/top-genres/christian/",
"http://www.songlyrics.com/news/top-genres/country-music/",
"http://www.songlyrics.com/news/top-genres/hip-hop-rap/",
"http://www.songlyrics.com/news/top-genres/rhythm-blues/",
"http://www.songlyrics.com/news/top-genres/pop/",
"http://www.songlyrics.com/news/top-genres/rock/"]

for i in range(0,6):
    time.sleep(5)
    doc = get( urllist[i] ).text
    soup = BeautifulSoup(doc, 'html.parser')
    div = soup.find( 'div', { 'class': 'box listbox' } )

# get genres
    title = soup.title.get_text().split(' ')

    index100 = title.index('100')
    indexSongs = title.index('Songs')
    genre = ' '.join(title[(index100+1):(indexSongs)]).encode('utf-8')

# create list of top 100 songs by genre
    songs = div.find_all('a')
    songlinks = []

# create loop to extract song links
    for j in range(0,200): #[0::2]:
        time.sleep(1)
        songlink = songs[j].get('href')
        songlinks.append(songlink) #output links to a list called songlinks

    songlinks = filter(None, songlinks)
    songlinks = [songlink for songlink in songlinks if (len(songlink.split('/'))==6)]

    for k in range(0,len(songlinks)):
        songdoc = get( songlinks[k] ).text
        songsoup = BeautifulSoup(songdoc, 'html.parser')
        songinfo = songsoup.title.get_text()
    #    print(songinfo, 'is number', k)

        songdiv = songsoup.find( 'div', { 'id': 'songLyricsDiv-outer' } )
        if songdiv != None:
            lyrics = songdiv.getText().replace("\n", " ").replace("\'", "").replace("\r", " ").encode('utf-8')

            lyricsvector.append(lyrics)
            genrevector.append(genre)
            songinfovector.append(songinfo)
            #writer.writerow((lyrics,genre,songinfo))




with open('newExcel.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(("Lyrics","Genre","song"))

    for i in range(len(lyricsvector)):
        writer.writerow((lyricsvector[i],genrevector[i],songinfovector[i]))
