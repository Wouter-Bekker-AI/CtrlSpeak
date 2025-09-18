import wave
from CtrlSpeak import transcribe_audio

def split_wav(input_file, split_time, output_file1, output_file2):
    # Open the original WAV file
    with wave.open(input_file, 'rb') as wav:
        # Extract parameters
        params = wav.getparams()
        framerate = wav.getframerate()
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()

        # Calculate the split point in frames
        split_frame = int(framerate * split_time)

        # Read frames up to the split point
        wav.setpos(0)
        frames_part1 = wav.readframes(split_frame)

        # Read frames from the split point to the end
        frames_part2 = wav.readframes(wav.getnframes() - split_frame)

    # Write the first part to a new WAV file
    with wave.open(output_file1, 'wb') as wav_part1:
        wav_part1.setparams((n_channels, sampwidth, framerate, split_frame, params.comptype, params.compname))
        wav_part1.writeframes(frames_part1)

    # Write the second part to another new WAV file
    with wave.open(output_file2, 'wb') as wav_part2:
        wav_part2.setparams((n_channels, sampwidth, framerate, len(frames_part2) // (n_channels * sampwidth), params.comptype, params.compname))
        wav_part2.writeframes(frames_part2)

# Example usage:
split_wav('temp - Copy.wav', 180, 'part1.wav', 'part2.wav')

print(transcribe_audio('part1.wav'))
print(transcribe_audio('part2.wav'))