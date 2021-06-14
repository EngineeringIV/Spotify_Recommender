# Spotify_Recommender
A recommendation engine to recommend K-pop songs based on Spotify Kpop playlist

 ## Analysis Motivation
I have to admit that I am a huge K-pop fan (don't judge). There are thousands of K-pop songs that are released each year and it is difficult to keep track of every single of them. 
Fortunately, streaming services like Spotify made it a lot easier to listen to similar songs based on your preference and the playlists you've created. In this project, I wanted 
to take a crack of the Spotify recommendation engine and build a recommendation engine based on playlists I've created in my personal Spotify account. I've created a playlist
called "KPOP" in my Spotify account and added popular songs from BTS (of course they have to be here lol), Blackpink, and my personal favorite IU and I'd like to be able to
recommend the top 30 songs (hopefully kpop songs) based on this playlist. 

## Design approach
Since I don't have access to many other user's preferences, I chose to design a content-based recommendation engine solely based on songs' attributes. I used the [Kaggle Spotify Database](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks)
which is consist of 160k songs from 1922 to 2021 with many attributes such as artist, genre, popularity, release year and more technical attributes such as key, loudness, and tempo. 

I converted genre (a categorical feature) to numerical feature with One-Hot-Encoding and grouped popularity into different buckets. I then used a TF-IDF to fit and transform the genres
in order to extract more relevant information. After these conversion, I appended atrributes besides genre and popularity in to a new dataframe so that each song has a processed feature
vector.

Then, I leveraged the Spotify Web API called [Spotipy](https://spotipy.readthedocs.io/en/2.16.1/) and the [Spotify for Developers](https://developer.spotify.com/) to access my own playlist
and extract all the songs with their IDs. Then, I extracted all the attributes for the playlist songs and summarized into a single row vector to represent the playlist. To account
for change of song tastes overtime, I added a time weight to penalize songs added long time ago.

Finally, I calculated the cosine similarity between the playlist summarized vector with individual songs and used the cosine similarity score to make my recommendations. 

## File Structure
- Python files
  - main.py: main execution file

- Data files (In "Data" folder)
  - data_o.csv: complete dataset, audio features of tracks
  - data_by_artist_o.csv: audio features of artists
  - data_by_genre_o.csv: audio features of genres (not used in the code)
  - data_by_year_o.csv: audio features of years (not used in the code)
  - artists.csv: popularity metrics of artists (not used in the code)
  - tracks.csv: expanded list of artists (not used in the code)

- Txt file
  - requirements.txt: packages required to run the code

## Analysis Results
After running the recommendation engine on my Kpop playlist, the following result were generated. As you can see, the recommendation successfully recommended other songs by ***BlackPink***
(Bet You Wanna, Crazy Over You Love To Hate Me, Sour Candy, You Never Know, Forever Young, Really, DDU-DU DDU-DU, See U Later, Kill This Love, Don't Know What To Do, As If It's your Last) and some popular songs from other K-pop group such as ***Twice*** (More & More, Yes Or Yes, Dance The Night Away, What is Love, Feel Special, I CAN'T STOP ME, Why Not?), 
***Mamamoo*** (Dingga, AYA, Egotistic, Hip, Gogobebe), ***GFriend*** (MAGO), ***Everglow*** (LA DI DA, Bon Bon Chocolat, Adios), ***Momoland*** (Boom Boom), and ***Loona*** (Hi High). All 30 songs recommended are indeed K-pop song!
![Image of result](https://github.com/EngineeringIV/Spotify_Recommender/blob/main/Figure_1.png)


## Credit and Acknowledgement
This project follows the wonderful [tutorial](https://www.youtube.com/watch?v=tooddaC14q4) by [Madhav Thaker](https://www.youtube.com/channel/UC0-S_HnWTDFaXgTbYSL46Ug). 
It also uses [Spotipy](https://spotipy.readthedocs.io/en/2.16.1/), [Spotify for Developers](https://developer.spotify.com/), and [Kaggle Spotify Database](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks). Lastly, many thanks to [Spotify](https://www.spotify.com/us/home/) for creating a wonderful recommendation engine to recommend great songs and also creating a great tool and API for developers. 

