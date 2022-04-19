import csv, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

bg_color = '#fff'

path_font = os.path.join('font', 'NotoSansTC-Regular.otf')
title_font = FontProperties(fname=path_font, size=48)
tick_font = FontProperties(fname=path_font, size=20)
label_font = FontProperties(fname=path_font, size=28)

x = [0,0.1,0.3,0.5,1.0]
y = []
stds = []
with open('results/data.csv', 'r', encoding='utf-8') as file:
    rows = csv.reader(file)
    for row in zip(*rows):
        row = [float(i) for i in row if i]
        y.append(np.mean(row))
        stds.append(np.std(row))

plt.figure(figsize=(16,10), dpi=600, facecolor=bg_color)
plt.title('以各食鹽水濃度種植綠豆大小', fontproperties=title_font)
plt.errorbar(x, y, yerr=stds, ecolor='k', linewidth=4, capsize=10)

plt.xticks(x, fontproperties=tick_font)
plt.yticks(np.arange(0.5, 2.5, 0.25), fontproperties=tick_font)

plt.xlabel('食鹽水重量百分比濃度(%)', fontproperties=label_font)
plt.ylabel('綠豆大小(平方公分)', fontproperties=label_font)

for x1, y1 in zip(x, y):
    print(x1, y1)
    posX, posY = x1, y1
    if posX < 1:
        posX += 0.005
    else:
        posX = 0.93
    posY += 0.02
    plt.text(posX, posY, f'{y1:.3f}', fontproperties=tick_font)

plt.savefig('results/line_chart.jpg')