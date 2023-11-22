import os
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import speedup

def create_mp3_file(filename, words, times):
    audio_dur = times[-1][1]
    
    # Make a bunch of mp3 files
    ## final_audio = AudioSegment.silent(duration=audio_dur)
    final_audio = AudioSegment.silent(duration=100)
    
    mp3_ind = 1

    previous_word_end_time = 0
    for word, time in zip(words, times):
        assert(time[1] > time[0])
        # Text to speech the word..
        tts = gTTS(word, lang='en', slow=False)  # slow=False means the speed is not slow
        output_file = f"{mp3_ind}.mp3" #
        tts.save(output_file)
        
        # # Speed up or slow down the word
        # desired_length_ms = (time[1] - time[0]) * 1000
        
        sound = AudioSegment.from_file(output_file, format="mp3")
        # current_length_ms = len(sound)
        # new_speed = current_length_ms / desired_length_ms
        # print(new_speed)
        new_speed = 1
        sound = speedup(sound, playback_speed=new_speed)

        # # First add silence from last word
        # silence_dur = (time[0] * 1000) - previous_word_end_time
        # if silence_dur > 0:
        #     final_audio = final_audio.append(AudioSegment.silent(duration=silence_dur))
        
        final_audio = final_audio.append(sound)
        mp3_ind += 1

        previous_word_end_time = time[1] * 1000
    
    final_audio.export(filename, format="mp3")
    return final_audio
        
        



if __name__ == "__main__":
    os.makedirs("./mp3_files", exist_ok=True)
    
    words = "This is a test of the timing system".split()
    times = [(0.1, 0.3), (0.3, 0.5), (1, 1.4), (1.4, 2.5), (2.6, 2.9), (3.2, 4.0), (4.1, 5.0), (6.9, 7.11)]
    
    create_mp3_file("final_audio_test.mp3", words, times)