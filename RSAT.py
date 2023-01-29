#Copyright (c) 2023 Kotetsu0000
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import sys
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from queue import Queue
import threading
import traceback
import time

from espnet2.bin.asr_inference import Speech2Text
import soundcard as sc
import torch
import numpy as np

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack(expand=1, fill=tk.BOTH, anchor=tk.NW)
        ##ここまでTkinterテンプレ##
        self.master=master

        #スピーカーリスト
        self.speaker_list = [str(i.name) for i in sc.all_speakers()]
        self.speaker_default = str(sc.default_speaker().name)

        self.q = Queue()

        #ウィジェットの生成
        self.create_widgets()

        self.load_thread = threading.Thread(target=self.set_model_auto, daemon=True)
        self.load_thread.start()

    def create_widgets(self):
        #全体領域の定義
        self.main_window = tk.PanedWindow(self, orient="vertical")
        self.main_window.pack(expand=True, fill=tk.BOTH, side=tk.TOP)

        #GPUが使用できるかの表示、モデルの準備完了に関しての表示
        self.set_window = tk.PanedWindow(self.main_window, orient="vertical")
        self.set_window.pack(fill=tk.BOTH, side=tk.TOP)

        #音声入力選択
        self.speaker_window = tk.PanedWindow(self.main_window, orient="vertical")
        self.speaker_window.pack(fill=tk.BOTH, side=tk.TOP)

        #ログ表示用領域
        self.log_window = tk.PanedWindow(self.main_window, orient="vertical")
        self.log_window.pack(fill=tk.BOTH, side=tk.TOP, expand=True)


        #GPUの使用ができるかのチェック
        self.GPU_ok = tk.BooleanVar()
        self.GPU_ok.set(torch.cuda.is_available())
        text = 'GPU使用可能' if torch.cuda.is_available() else 'GPU使用不可'
        self.GPU_check_box = tk.Checkbutton(self.set_window, variable=self.GPU_ok, state='disabled')
        self.GPU_check_box.pack(fill=tk.BOTH, side=tk.LEFT)
        self.GPU_use_label = ttk.Label(self.set_window, text=text)
        self.GPU_use_label.pack(fill=tk.BOTH, side=tk.LEFT)

        #モデルのロード確認
        self.model_ok = tk.BooleanVar()
        self.model_ok.set(False)
        self.model_check_box = tk.Checkbutton(self.set_window, variable=self.model_ok, state='disabled')
        self.model_check_box.pack(fill=tk.BOTH, side=tk.LEFT)
        self.model_use = tk.StringVar(value='モデルロード中...')
        self.model_use_label = ttk.Label(self.set_window, textvariable=self.model_use)
        self.model_use_label.pack(fill=tk.BOTH, side=tk.LEFT)

        #ログの削除ボタン
        self.log_delete_button = tk.Button(self.set_window, text='ログのクリア', command=self.clear_log)
        self.log_delete_button.pack(fill=tk.BOTH, side=tk.RIGHT)

        #モデルのロードに関して
        self.select_speaker = tk.StringVar(value=self.speaker_default)
        self.speaker_combobox = ttk.Combobox(self.speaker_window, textvariable= self.select_speaker, values=self.speaker_list, state="readonly")
        self.speaker_combobox.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
        self.speaker_combobox.bind('<<ComboboxSelected>>', self.change_speaker)

        #ログ部分
        self.log_space = scrolledtext.ScrolledText(self.log_window, state='disabled')
        self.log_space.pack(fill=tk.BOTH, side=tk.TOP, expand=True)

    def change_speaker(self,event):
        self.change_speaker_bool = False

    def add_log(self, text):
        self.log_space.config(state='normal')
        self.log_space.insert(tk.END,text)
        self.log_space.insert(tk.END,'\n')
        self.log_space.config(state='disabled')
        self.log_space.see("end")

    def clear_log(self):
        self.log_space.config(state='normal')
        self.log_space.delete(0.,tk.END)
        self.log_space.config(state='disabled')

    def set_model_auto(self):
        try:
            device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
            self.speech2text = Speech2Text.from_pretrained(
                asr_train_config=temp_path('./exp/asr_train_asr_conformer_raw_jp_char/config.yaml'), 
                lm_train_config=temp_path('./exp/lm_train_lm_jp_char/config.yaml'), 
                asr_model_file=temp_path('./exp/asr_train_asr_conformer_raw_jp_char/valid.acc.ave_10best.pth'), 
                lm_file=temp_path('./exp/lm_train_lm_jp_char/40epoch.pth'),
                beam_size=5,
                batch_size=0,
                device=device
                )

            self.model_use.set('モデル起動中')
            self.model_ok.set(True)

            self.recording_thread = threading.Thread(target=self.recording, daemon=True)
            self.recognize_thread = threading.Thread(target=self.recognize, daemon=True)
            self.recording_thread.start()
            self.recognize_thread.start()
        except:
            messagebox.showerror('エラー', traceback.format_exc())
            self.master.destroy()

    def recognize(self):
        try:
            while True:
                if self.model_ok.get():
                    audio = self.q.get()
                    if (audio ** 2).max() > 0.0001:
                        result = self.speech2text(audio)[0][0]

                        # print the recognized text
                        self.add_log(f'{result}')
                else:time.sleep(1)
        except:
            messagebox.showerror('エラー', traceback.format_exc())
            self.master.destroy()

    def recording(self):
        try:
            SAMPLE_RATE = 16000
            INTERVAL = 3
            BUFFER_SIZE = 4096
            b = np.ones(100) / 100

            audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
            n = 0
            while True:
                self.change_speaker_bool=True
                print(self.select_speaker.get())
                with sc.get_microphone(id=str(self.select_speaker.get()), include_loopback=True).recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
                    while self.change_speaker_bool:
                        while n < SAMPLE_RATE * INTERVAL:
                            data = mic.record(BUFFER_SIZE)
                            audio[n:n+len(data)] = data.reshape(-1)
                            n += len(data)

                        # find silent periods
                        m = n * 4 // 5
                        vol = np.convolve(audio[m:n] ** 2, b, 'same')
                        m += vol.argmin()
                        self.q.put(audio[:m])

                        audio_prev = audio
                        audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
                        audio[:n-m] = audio_prev[m:n]
                        n = n-m
        except:
            messagebox.showerror('エラー', traceback.format_exc())
            self.master.destroy()

def temp_path(relative_path):
    try:
        #Retrieve Temp Path
        base_path = sys._MEIPASS
    except Exception:
        #Retrieve Current Path Then Error 
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

if __name__ == "__main__":
    root = tk.Tk()
    w = root.winfo_screenwidth()    #モニター横幅取得
    h = root.winfo_screenheight()   #モニター縦幅取得
    windw_w = 700
    windw_h = 400

    root.title(u"ReazonSpeech Auto Transcriber")
    iconfile = 'icon.ico'
    icon = temp_path(iconfile)
    root.iconbitmap(default=icon)
    root.geometry(str(windw_w)+"x"+str(windw_h)+"+"+str((w-windw_w)//2)+"+"+str((h-windw_h)//2))
    app = Application(master=root)
    app.mainloop()
# --add-data ".env/Lib/site-packages/wandb/*;./wandb/" --hidden-import wandb --hidden-import wandb_gql 