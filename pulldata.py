import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

# Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id='da3d78cadd564ac0b97bbe178dbc5c65',
    client_secret='f4c8bfdad02c4b41a08cd03bf49aa756',
    redirect_uri='http://localhost:8000/callback',
    scope='user-library-read'
))

def get_track_features(track_id):
    features = sp.audio_features(track_id)[0]
    if features:
        return {
            'id': track_id,
            'danceability': features['danceability'],
            'energy': features['energy'],
            'key': features['key'],
            'loudness': features['loudness'],
            'mode': features['mode'],
            'speechiness': features['speechiness'],
            'acousticness': features['acousticness'],
            'instrumentalness': features['instrumentalness'],
            'liveness': features['liveness'],
            'valence': features['valence'],
            'tempo': features['tempo'],
            'time_signature': features['time_signature']
        }
    return None

def fetch_liked_songs():
    tracks = []
    results = sp.current_user_saved_tracks()
    while results:
        tracks.extend(results['items'])
        # Check if there's a next page
        if results['next']:
            results = sp.next(results)
        else:
            break
    
    track_ids = [track['track']['id'] for track in tracks]
    track_features = [get_track_features(track_id) for track_id in track_ids]
    return track_features

# Fetch liked songs
track_features = fetch_liked_songs()

# Convert to DataFrame
df = pd.DataFrame(track_features)

# Save to CSV
df.to_csv('liked_songs_features.csv', index=False)

print("Liked songs features have been saved to liked_songs_features.csv")
