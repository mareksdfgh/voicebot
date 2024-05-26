import pygame
import gtts
import tempfile
import speech_recognition as sr
import ollama
import eel
from faster_whisper import WhisperModel
from threading import Thread

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Initialize WhisperModel
model_size = "medium"  # Oder eine andere gewünschte Modellgröße
model = WhisperModel(model_size, device="cpu", compute_type="float32")

messages = []

# Variable to store the state of speech recognition
speech_recognition_on = False

def update_speech_recognition_status(status):
    global speech_recognition_on
    speech_recognition_on = status

def send(chat):
    stream_messages = messages[:]  # Kopie der Nachrichtenliste erstellen, um die erste Nachricht zu entfernen
    if stream_messages:
        stream_messages.pop(0)  # Entferne die erste Nachricht aus der Liste
    stream_messages.append(
        {
            'role': 'user',
            'content': chat,
        }
    )
    stream = ollama.chat(model='llama3',
                         messages=stream_messages,
                         stream=True,
    )

    response = ""
    for chunk in stream:
        part = chunk['message']['content']
        print(part, end='', flush=True)
        response = response + part

    if response.strip():  # Überprüfen, ob der Text nicht leer ist
        messages.append(
            {
                'role': 'assistant',
                'content': response,
            }
        )

        tts = gtts.gTTS(text=response, lang='en', slow=False)

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            tts.save(f.name)
            f.close()  
            pygame.mixer.music.load(f.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
    else:
        print("No response received")

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        update_speech_recognition_status(True)  
        audio = recognizer.listen(source)
        update_speech_recognition_status(False)  

    try:
        print("Recognizing...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio.get_wav_data())
            audio_path = f.name

        segments, info = model.transcribe(audio_path, beam_size=5, language="en")
        recognized_text = ' '.join([segment.text for segment in segments])

        print("You said:", recognized_text)
        send(recognized_text)
    except Exception as e:
        print("Sorry, an error occurred:", e)

# Predefined initial message from the user
initial_user_message = ""

send(initial_user_message)

conversation_file = "conversation.txt"

def save_conversation():
    with open(conversation_file, "a+") as file:
        file.seek(0)  
        existing_content = file.read()  
        file.seek(0, 0)  
        for message in messages:
            role = message['role']
            content = message['content']
            file.write(f"{role}: {content}\n")
        file.write(existing_content)  

def start_listening():
    while True:
        listen()
        save_conversation()

listen_thread = Thread(target=start_listening)
listen_thread.start()

# Set web files folder and optionally specify which file types to check for eel.expose()
eel.init('web')

# Expose Python functions to JavaScript
@eel.expose
def get_speech_recognition_status():
    return {'status': speech_recognition_on}

# Start the Eel app
eel.start('index.html', size=(300, 150))
