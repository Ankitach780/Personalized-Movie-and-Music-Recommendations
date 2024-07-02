import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from keras.models import model_from_json
json_file=open("emotiondet.json","r")
model_json=json_file.read()
json_file.close()
model=model_from_json(model_json)
model.load_weights("emotiondet.h5")

movies=joblib.load(open("movie.pkl",'rb'))
music=joblib.load(open('music.pkl','rb'))

labels = ['Angry', 'Disguist', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

emotion_to_genre = {
    'Sad': 'Drama',
    'Disguist': 'Crime',
    'Angry': 'Action',
    'Fear': 'Horror',
    'Happy': 'Comedy',
    'Neutral': 'Family',
    'Surprise': 'Fantasy'
}
def recommend_movies(Emotion):
    genre = emotion_to_genre.get(Emotion)
    if genre:
        filter = movies[movies['Genre'].str.contains(genre, case=False, na=False)]
        return filter ['Movie_Name'].head(2).values if not filter.empty else ["No movies found"]
    else:
        return ["No genre found for the emotion"]

def recommend_music(emotion):
    filter=music[music['Emotion']==emotion]
    return filter['Name'].head(2).values if not filter.empty else ["No music found"]
    

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (64, 64))
        
        img = gray_resized.reshape(1, 64, 64, 1)
        img = img.astype('float32')/255.0
        
        prediction = model.predict(img)
        predicted_label = labels[np.argmax(prediction)]
        movie_rec=recommend_movies(predicted_label)
        music_rec=recommend_music(predicted_label)
        y_offset = 100
        cv2.putText(frame, "Movies:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 25, 0), 2)
        for i, movie in enumerate(movie_rec):
            y_offset += 30
            cv2.putText(frame, movie, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (23, 25, 0), 2)
        
        y_offset += 90 
        cv2.putText(frame, "Music:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 25, 0), 2)
        for i, music in enumerate(music_rec):
            y_offset += 30
            cv2.putText(frame, music, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (23, 25, 0), 2)
        
        cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 25, 0), 2)
        
        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
