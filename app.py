from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
import librosa
import IPython.display as ipd


model, df_state, _ = init_df()  # Load default model
#enhanced_audio = enhance(model, df_state, noisy_audio)

audio_path="noisy_snr0.wav"
audio, _ = load_audio(audio_path, sr=df_state.sr())

# Faylni yuklab olish
audio_file = 'noisy_snr0.wav'
a, sr = librosa.load(audio_file)

# Faylni ijro etish
ipd.Audio(a, rate=sr)

enhanced = enhance(model, df_state, audio)
# Save for listening
save_audio("enhanced.wav", enhanced, df_state.sr())

