# 輕鬆用 Intel AI PC 及 OpenVINO 建立數位分身 （digital_avatar）
by Jack OmniXRI, 2025/03/24  

這裡的範例是基於 **Intel OpenVINO GenAI** 及 **Notebooks 2025.0** 在 Khadas Mind 2 AI Maker Kit (**Intel AI PC Core Ultra 7 258V**)上進行測試，主要整合下列內容：  
* 語音轉文字(Speech to Text, STT) - whisper-asr-genai https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/whisper-asr-genai  
* 大語言模型對話(Large Language Model, LLM) - deepseek-r1 https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/deepseek-r1  

## 主要檔案說明：  
* whisper-asr-genai_run.ipynb 語音轉文字 Jupyter Notebook 範例  
* gradio_helper_whisper_run.py 語音轉文字 Gradio 網頁人機介面  
* deepseek-r1_run.ipynb 大語言模型 Jupyter Notebook 範例  
* gradio_helper_deepseek_run.py 大語言模型 Gradio 網頁人機介面  
* llm_config_run.py 大語言模型相關參數設置  
* digital_avatar_run.ipynb 數字分身 Jupyter Notebook 範例  
* gradio_helper_digital_avatar_run.py 數字分身 Gradio 網頁人機介面  
* 20250324_Digital_Avatar_07.jpg 為範例執行後 Gradio 介面顯示結果

**更完整的文章說明可參考 [輕鬆用 Intel AI PC 及 OpenVINO 建立數位分身](https://hackmd.io/@OmniXRI-Jack/digital_avatar)**  

![20241212_Digital_Human_Fig04.jpg](https://raw.githubusercontent.com/OmniXRI/digital_avatar/refs/heads/main/20250324_Digital_Avatar_07.jpg)

## 範例說明

**digital_avatar_run.ipynb* 整合 **Intel OpenVINO GenAI** 及 **Notebooks 2025.0** 中的 Whisper STT 及 DeepSeek-R1 LLM 範例，完成一個語音輸入問題，文字輸出到大語言模型，再以文字輸出答案，達成一個簡單版的數位分身前半段。所有操作介面採Gradio。執行前至少要先執行一次兩個原始範例。

STT 原始範例: https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/whisper-asr-genai/whisper-asr-genai.ipynb  

原始範例全部執行(Restart Kernel and Run All Cells）後，預設會取得 whisper-tiny 模型， 經轉換後會得到 whisper-tiny-quantized 量化後較小的模型。如果辨識率不佳想要使用大一點的模型，可採用手動單步執行方式完成。本範例為簡化版，執行前需先完整執行過原始範例，取得模型並轉換好 OpenVINO 所需 IR 格式。  

LLM 原始範例: https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/deepseek-r1/deepseek-r1.ipynb  

原始範例全部執行(Restart Kernel and Run All Cells）後，預設會取得 DeepSeek-R1-Distill-Qwen-14B 模型， INT4 權重壓縮格式，預計下載約 15 GB，轉檔後會產生 7.81 GB 大小檔案。如想要使用小一點的模型，可採用手動單步執行方式完成。本範例為簡化版，執行前需先完整執行過原始範例，取得模型並轉換好 OpenVINO 所需 IR 格式。  

待上述動作完成，即可執行本範例。這裡使用 Gradio 作為操作介面，預設啟動網址為 http://127.0.0.1:7869/  (http://localhost:7869/ ，操作步驟如下：  
1. 在 Whisper 語音轉文字區上傳或錄製一段聲音檔案，中文或中英文混合皆可。  
2. 按下「Transcribe」開始轉換，結果會顯示在輸出文字區。  
3. 將文字內容複製到 DeepSeek-R1 大語言模型輸入問題區。  
4. 按下「Summit」即可開始對話。  
5. 若發生對話產生到一半就停止，可點擊進階選項(Adavance Options)，將最大新詞元數(Max New Tokens)調至最大即可改善。
6. 若不想顯示簡體中文可加入提示句「以下內容請以繁體中文顯示」即可。

## 延伸閱讀： 
* [【vMaker Edge AI專欄 #24】 如何使用 Gradio 快速搭建人工智慧應用圖形化人機介面](https://omnixri.blogspot.com/2024/12/vmaker-edge-ai-24-gradio.html)
* [如何使用 Intel AI PC 及 OpenVINO 實現虛擬主播](https://hackmd.io/@OmniXRI-Jack/Digital-Human)
