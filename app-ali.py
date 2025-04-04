import pyaudio  
import wave  
#import numpy as np  
import time
#import requests  
import json
from dotenv import load_dotenv
import os  
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
import requests
import time
from pydub import AudioSegment  

load_dotenv("./qwen.env")  
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.environ["key"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

 
# 配置  
FORMAT = pyaudio.paInt16  # 音频格式  
CHANNELS = 1               # 单声道  
RATE = 44100               # 采样率  
CHUNK = 1024               # 每次读取的音频帧数  
THRESHOLD =10000            # 音量阈值  
RECORD_SECONDS = 20        # 最大录音时间  
SILENCE_DURATION = 500      # 静音持续时间（秒） 
sk_key=os.getenv('key') 

def get_volume(data):  
    """计算音量"""  
    return np.linalg.norm(np.frombuffer(data, dtype=np.int16)  )
  
def record_audio():  
    """录制音频并保存为 WAV 文件"""  
    audio = pyaudio.PyAudio()  
      
    # 开始流  
    stream = audio.open(format=FORMAT, channels=CHANNELS,  
                        rate=RATE, input=True,  
                        frames_per_buffer=CHUNK)  
      
    print("\n请讲...")  
  
    frames = []  
    recording = False  
    silence_start_time = None  # 用于记录静音开始时间  
    start_time = time.time()    # 记录开始时间  
  
    while True:  
        # 读取音频数据  
        data = stream.read(CHUNK)  
        volume = get_volume(data)  
        if volume>10000:
            print(volume)  
        if abs(volume) > THRESHOLD and not recording:  
            print("AI: 持续聆听中...")  
            recording = True  
            start_time = time.time()  # 记录开始时间  
            silence_start_time = None  # 重置静音计时器  
          
        if recording:  
            frames.append(data)  
            #print(f"录音中... 音量: {volume}")  
            #print(f".",end="")  
  
            # 检查录音时间  
            if time.time() - start_time > RECORD_SECONDS:  
                print("达到最大录音时间，停止录音")  
                break  
            if time.time()*1000-(start_time*1000)<2000:
                continue
            # 检查静音  
            if abs(volume) < THRESHOLD:
                if silence_start_time is None:
                    silence_start_time=time.time()*1000
                diff = time.time()*1000 - silence_start_time
                if diff > SILENCE_DURATION:  
                    print(f"检测到静音[{diff}ms]超过 1 秒，停止录音")  
                    break 
            else:  
                silence_start_time = None # 记录静音开始时间  
  
    # 停止流  
    stream.stop_stream()  
    stream.close()  
    audio.terminate()  
  
    # 保存录音  
    if frames:  
        with wave.open("output.wav", 'wb') as wf:  
            wf.setnchannels(CHANNELS)  
            wf.setsampwidth(audio.get_sample_size(FORMAT))  
            wf.setframerate(RATE)  
            wf.writeframes(b''.join(frames))  
        print("录音已保存为 output.wav")  
        
    else:  
        print("没有录音数据")  



sys_msg={
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            }
messages=[]
def generate():  
    global messages  
    base64_audio = encode_audio("output.wav")
    messages=messages[-5:]+[{
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:;base64,{base64_audio}",
                            "format": "wav",
                        },
                    }
                ],
            }]
    completion = client.chat.completions.create(
        model="qwen2.5-omni-7b",
        messages=[sys_msg]+messages[-6:],
        # 设置输出数据的模态，当前支持两种：["text","audio"]、["text"]
        modalities=["text", "audio"],
        audio={"voice": "Chelsie", "format": "wav"},
        # stream 必须设置为 True，否则会报错
        stream=True,
        stream_options={"include_usage": True},
    )

    p = pyaudio.PyAudio()
    #创建音频流
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,
                    output=True)
    result_all=""
    print("AI:")
    for chunk in completion:
        if chunk.choices:
            if hasattr(chunk.choices[0].delta, "audio"):
                try:
                    delta = chunk.choices[0].delta
                    #print(delta)
                    transcript = delta.audio.get("transcript")
                    audio_string = delta.audio.get("data")
                    if hasattr(delta,'audio') and delta.audio and transcript:
                        print(transcript,end="")
                        result_all=result_all+transcript
                    if hasattr(delta,'audio') and delta.audio and audio_string:
                        wav_bytes = base64.b64decode(audio_string)
                        audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
                        #直接播放音频数据
                        stream.write(audio_np.tobytes())
                except Exception as e:
                    print(f"error:{e}")
                    
                    
    
    time.sleep(0.8)
    #清理资源
    stream.stop_stream()
    stream.close()
    p.terminate()
    messages.append({"role": "assistant", "content": result_all.strip()})  
    


if __name__ == "__main__":
    while True:
        record_audio()
        generate()
        