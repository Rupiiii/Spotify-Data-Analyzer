import cv2
import numpy as np
from tensorflow.keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random

# Spotify credentials (replace these with your credentials)
SPOTIPY_CLIENT_ID = ''
SPOTIPY_CLIENT_SECRET = ''
SPOTIPY_REDIRECT_URI = 'http://localhost:8000/callback'

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope="playlist-modify-public playlist-modify-private user-library-read user-library-modify"))

# Load the saved emotion recognition model
try:
    model = load_model('fine_tuned_mood_recognition_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the emotion labels
emotion_labels = ['happy', 'calm', 'sad', 'angry', 'surprised']

# Mapping for swapping labels
emotion_swap = {
    'happy': 'sad',
    'sad': 'happy',
    'calm': 'calm',  # No change for calm
    'angry': 'angry',  # No change for angry
    'surprised': 'surprised'  # No change for surprised
}

# Open the webcam for real-time emotion detection
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

print("Webcam opened successfully.")

last_detected_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected.")
    
    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Draw a purple rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 0, 128), 2)

        # Preprocess the detected face for the emotion recognition model
        face = frame[y:y + h, x:x + w]  # Crop the face
        face = cv2.resize(face, (150, 150))  # Resize to match model input
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = face / 255.0  # Normalize

        # Get predictions
        try:
            preds = model.predict(face)
            pred_index = np.argmax(preds)
        except Exception as e:
            print(f"Error during prediction: {e}")
            pred_index = -1

        # Print raw predictions for debugging
        print(f"Raw Predictions: {preds}")
        print(f"Predicted Index: {pred_index}")

        # Ensure the prediction is within the valid index range
        if 0 <= pred_index < len(emotion_labels):
            emotion_label = emotion_labels[pred_index]
            # Apply the label swap
            emotion_label = emotion_swap.get(emotion_label, emotion_label)
        else:
            print(f"Invalid prediction index: {pred_index}")
            emotion_label = "unknown"

        # Update last detected emotion
        last_detected_emotion = emotion_label

        # Display the emotion on the frame near the face
        cv2.putText(frame, f"Emotion: {emotion_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the detected face and emotion label
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# After the loop, fetch Spotify songs based on the detected mood
def recommend_songs(emotion):
    # Set mood-based search parameters
    if emotion == 'happy':
        query = 'happy'
    elif emotion == 'calm':
        query = 'calm'
    elif emotion == 'sad':
        query = 'sad'
    elif emotion == 'angry':
        query = 'angry'
    else:
        query = 'surprise'
    
    # Search for top tracks based on the mood
    try:
        results = sp.search(q=query, limit=50, type='track')  # Fetch more results to ensure a broader selection
        tracks = results['tracks']['items']
    except Exception as e:
        print(f"Error during Spotify search: {e}")
        return
    
    # Initialize list to track seen songs
    unique_tracks = []
    seen_titles = set()
    
    # Filter out repeated songs based on title and artist
    for track in tracks:
        track_title = track['name']
        track_artist = track['artists'][0]['name']
        if (track_title, track_artist) not in seen_titles:
            unique_tracks.append(track)
            seen_titles.add((track_title, track_artist))
    
    # Ensure there are enough unique tracks
    if len(unique_tracks) < 5:
        print("Not enough unique tracks found.")
        return
    
    # Shuffle the unique tracks to randomize the order
    random.shuffle(unique_tracks)
    
    # Select the top 5 tracks from the shuffled list
    top_tracks = unique_tracks[:5]

    # Print the top 5 songs and their names
    print(f"\nTop 5 {emotion.capitalize()} Songs:")
    track_uris = []
    for i, track in enumerate(top_tracks):
        print(f"{i+1}. {track['name']} by {track['artists'][0]['name']}")
        track_uris.append(track['uri'])
    
    # Ask for feedback
    feedback = input(f"\nDo you want to add these songs to a new playlist for your mood '{emotion}'? (yes/no): ")
    if feedback.lower() == 'yes':
        # Create a new playlist
        try:
            user_id = sp.current_user()['id']
            playlist = sp.user_playlist_create(user_id, f"{emotion.capitalize()} Playlist", public=True)
            playlist_id = playlist['id']

            # Add the top songs to the playlist
            sp.playlist_add_items(playlist_id, track_uris)
            print("Songs have been added to your playlist!")

            # Generate and print the direct link to the playlist
            playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
            print(f"Here is the direct link to your playlist: {playlist_url}")
        except Exception as e:
            print(f"Error during playlist creation or song addition: {e}")
    else:
        print("No changes made to your playlists.")

# Call the recommendation function
if last_detected_emotion:
    recommend_songs(last_detected_emotion)
