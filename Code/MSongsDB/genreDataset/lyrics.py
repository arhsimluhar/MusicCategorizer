import requests
from bs4 import BeautifulSoup, Comment, NavigableString
import sys, codecs, json


def getLyrics(singer, song):
		#Replace spaces with _
		singer = singer.replace(' ', '_')
		song = song.replace(' ', '_')
		r = requests.get('http://lyrics.wikia.com/{0}:{1}'.format(singer,song))
		s = BeautifulSoup(r.text,'lxml')
		#Get main lyrics holder
		lyrics = s.find("div",{'class':'lyricbox'})
		if lyrics is None:
			#raise ValueError("Song or Singer does not exist or the API does not have Lyrics")
			return None
		#Remove Scripts
		[s.extract() for s in lyrics('script')]
     
		#Remove Comments
		comments = lyrics.findAll(text=lambda text:isinstance(text, Comment))
		[comment.extract() for comment in comments]

		#Remove unecessary tags
		for tag in ['div','i','b','a']:
			for match in lyrics.findAll(tag):
				match.replaceWithChildren()
		#Get output as a string and remove non unicode characters and replace <br> with newlines
		lyrics= str(lyrics).replace('\n','').replace('<br/>',' ')
		output=lyrics[22:-6:]
		try:
			return output
		except:
			return output.encode('utf-8')


genres={}
lyrics_found=0
import codecs
y=codecs.open("songLyrics1.txt","w")
line_num=0
with codecs.open('msd_genre_dataset.txt','r') as f:
	for line in f:
	    if line_num<=27432:
	    	pass
	    else:
	    	temp=line.split(',')
	    	#print "artist name=",temp[2],"Title=",temp[3]
	    	lyrics=getLyrics(temp[2],temp[3])
	    	if(lyrics!=None):
	    		if(lyrics.split(' ')[0]!='<span'):
	    			try:	
	    				genres[temp[0]]+=1
	    			except:
	    				genres[temp[0]]=1
	    			lyrics_found+=1
	    			y.write(temp[0]+'**/**'+lyrics)
	    			y.write('\n')
	    	if(line_num%100==0):
	    		print "At present progress\n"
	    		print "Songs processed: "+ str(line_num)
	    		print "Lyrics saved upto now:" +str(lyrics_found)
	    		for gen in genres.keys():
	    			print gen,":",genres[gen]
	    line_num+=1
y.close()