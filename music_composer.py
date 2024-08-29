import streamlit as st
import pretty_midi
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Preprocess MIDI Files
def preprocess_midi(file_path):
    if not os.path.exists(file_path):
        st.error(f"File {file_path} does not exist.")
        return None
    
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append([note.pitch, note.start, note.end])
        return np.array(notes)
    except EOFError:
        st.error("Error reading the MIDI file. The file might be corrupted or empty.")
        return None

# Build the LSTM Model
def build_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Generate Music Sequence
def generate_sequence(model, seed_sequence, num_notes):
    for _ in range(num_notes):
        prediction = model.predict(seed_sequence)
        seed_sequence = np.append(seed_sequence, prediction)
    return seed_sequence

# Convert sequence to MIDI
def sequence_to_midi(sequence):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for note in sequence:
        midi_note = pretty_midi.Note(velocity=100, pitch=int(note[0]), start=note[1], end=note[2])
        instrument.notes.append(midi_note)
    midi.instruments.append(instrument)
    return midi

# Streamlit Interface
st.title("AI Music Composer")
genre = st.selectbox("Select Genre", ["Classical", "Jazz", "Pop"])
tempo = st.slider("Tempo", 60, 180)
num_notes = st.slider("Number of Notes", 10, 100)

if st.button("Generate Music"):
    # Load and preprocess a MIDI file
    midi_data = preprocess_midi('data/BecauseOfYou.mid')
    
    if midi_data is not None:
        # Define input shape and output dimension
        input_shape = (midi_data.shape[1], 3)
        output_dim = 128

        # Build and load a trained model
        model = build_model(input_shape, output_dim)

        # Assume the model is already trained for simplicity
        seed_sequence = midi_data[:1]  # Use the first sequence as a seed

        generated_sequence = generate_sequence(model, seed_sequence, num_notes)

        midi_output = sequence_to_midi(generated_sequence)
        midi_data_bytes = midi_output.write()

        st.audio(midi_data_bytes)
        st.download_button("Download Music", data=midi_data_bytes, file_name="generated_music.mid")
