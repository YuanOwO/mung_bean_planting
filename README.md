# 110-2 生物探究與實作 食鹽水對綠豆生長的影響

## 簡介
我們分別採用了純水及各種濃度的食鹽水種植綠豆 5 天。我們的認為濃度愈高的食鹽水會使綠豆愈難生長，豆芽愈小，而實驗結果與我們的主張相符。

接著我們在 [finder.py](/finder.py) 利用 [OpenCV](https://opencv.org) 處理數據

先將圖片降低雜訊後，轉換至 HSV 色彩空間並加 3 個 channel 分離，我們發現豆芽的飽和度低於背景，且色相也與背景不同，進行二值化(`threshold`)來篩選豆芽(色相、飽和度的閾值分別設為 60及 70)，接著使用`cv2.findContours()`尋找豆芽的輪廓並計算其面積(像素)。

![plot](/results/0.1/0.1_plot.jpg)

將豆芽去除後，可將明度進行二值化(閾值設為 200)篩選出格線，再經由霍夫轉換(`cv2.HoughLinesP()`)來檢測直線，接著計算每個格線面積後取平均以得知 1 平方公分為多少像素，來換算豆芽的實際大小。

取得豆芽數據之後，刪去部分極端值並利用 Excel 進行 F 檢定驗證變異數是否相同，來判斷要使用同質性或異質性 t 檢定。

接著我們在 [data.py](/data.py) 使用 [matplotlib](https://matplotlib.org) 處理數據圖表，詳細數據請見[`results`](/results/)資料夾
![line chart](/results/line_chart.jpg)

## 其他
* 我們使用思源黑體(Google 稱 Noto Sans CJK)製作圖表的文字部分
* [`raw`](/raw/) 為存放原始照片的資料夾