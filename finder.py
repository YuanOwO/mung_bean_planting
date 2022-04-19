import csv, shutil, os, time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

##################################################
# Arguments
THRESHS = {
    'H': 60,
    'S': 70,
    'GRID': 200
}
HOUGH_ARGS = {
    'rho': 0.5,
    'theta': np.pi / 360,
    'threshold': 200,
    'minLineLength': 600,
    'maxLineGap': 400
}
COLORS = {
    'LINE': (0,0,255),
    'CONTOUR': (255,0,255),
    'PLOT': (255,255,255)
}
BEAN_MIN_SIZE = 500
##################################################
def join_path(*paths):
    return os.path.join(*paths)
# Configs
path_font = join_path('font', 'NotoSansTC-Regular.otf')
title_font = FontProperties(fname=path_font, size=48)
font = FontProperties(fname=path_font, size=28)
##################################################
start = time.time()
# 重建紀錄資料夾
try:
    shutil.rmtree('results')
except FileNotFoundError:
    pass
os.mkdir('results')
files = open(join_path('results', 'files.txt'), 'w', encoding='utf-8')
for dataname in os.listdir('raw'):
    # 去除副檔名(.jpg)
    dataname = dataname[:-4]
    print(f'processing {dataname}.jpg')
    #########################
    # 圖片儲存
    def write(filename, img, folder = ''):
        """將檔名加上資料夾位址，並寫入圖片"""
        if not filename.endswith('.jpg'):
            filename += '.jpg'
        filename = dataname+'_'+filename
        path_img = join_path('results', dataname, folder, filename)
        files.write(path_img+'\n')
        cv2.imwrite(path_img, img)
    # 建立數據資料夾
    path_root    = join_path('results', dataname)
    path_raw_img = join_path('raw', dataname+'.jpg')
    path_plot    = join_path(path_root, dataname+'_plot.jpg')
    path_csv     = join_path(path_root, dataname+'_results.csv')
    os.mkdir(path_root)
    os.mkdir(join_path(path_root, 'beans'))
    #########################
    img_raw = cv2.imread(path_raw_img)
    # 降噪
    img_blur = cv2.GaussianBlur(img_raw, (5,5), 0)
    kernel = np.ones((3,3), np.uint8)
    img_blur = cv2.dilate(img_blur, kernel)
    img_blur = cv2.erode(img_blur, kernel)
    write('raw', img_raw)
    write('blur', img_blur)
    del path_root, path_raw_img
    del kernel
    ####################
    # 傳換至 HSV
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[:,:, i] for i in [0,1,2]]
    _, thresh_h = cv2.threshold(h, THRESHS['H'], 255, cv2.THRESH_BINARY_INV)
    _, thresh_s = cv2.threshold(s, THRESHS['S'], 255, cv2.THRESH_BINARY_INV)
    # 尋找綠豆
    print('finding beans')
    mask_bean = cv2.add(thresh_h, thresh_s)
    mask_bean_inv = cv2.bitwise_not(mask_bean)
    contours_bean_all, _ = cv2.findContours(mask_bean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 
    img_bean_BGR = cv2.cvtColor(mask_bean_inv, cv2.COLOR_GRAY2BGR)
    img_bean_bg  = img_raw.copy()
    img_bean_all_BGR = img_bean_BGR.copy()
    img_bean_all_bg  = img_raw.copy()
    # 篩選
    contours_bean = []
    results = []
    print('\033[1A\r\033[2Kbeans:')
    for i, cnt in enumerate(contours_bean_all):
        area = cv2.contourArea(cnt)
        result = (i + 1, area, 0, 'PASS' if area < BEAN_MIN_SIZE else '')
        results.append(result)
        if (not i % 3) and i:
            print('\033[3A\r\033[J', end='')
        print('  #{:3}\t{:,.2f} px\t'.format(*result))
        if not result[-1]:
            contours_bean.append(cnt)
        img_one_bean_BGR = img_bean_BGR.copy()
        cv2.drawContours(img_one_bean_BGR, (cnt), -1, COLORS['CONTOUR'], 5)
        write(f'bean-{i+1}.jpg', img_one_bean_BGR, folder='beans')
    # 描繪輪廓
    cv2.drawContours(img_bean_all_BGR, contours_bean_all, -1, COLORS['CONTOUR'], 5)
    cv2.drawContours(img_bean_all_bg,  contours_bean_all, -1, COLORS['CONTOUR'], 5)
    cv2.drawContours(img_bean_BGR, contours_bean, -1, COLORS['CONTOUR'], 5)
    cv2.drawContours(img_bean_bg,  contours_bean, -1, COLORS['CONTOUR'], 5)
    # 儲存
    write('bean.jpg', img_bean_BGR)
    write('bean_background.jpg', img_bean_bg)
    write('bean_all.jpg', img_bean_all_BGR)
    write('bean_all_background.jpg', img_bean_all_bg)
    write('bean_mask', mask_bean)
    results.append((len(contours_bean_all), 'finded', len(contours_bean), 'filtered'))
    print('\033[{}A\r\033[J{} beans {}, and {} beans {}.'.format(
        len(contours_bean_all) % 3 + 1, *results[-1]))
    del contours_bean_all, contours_bean
    del img_blur, hsv, thresh_h, thresh_s
    del img_bean_all_BGR, img_bean_all_bg, img_bean_BGR, img_bean_bg
    #########################
    # 尋找格線
    print('finding gird')
    v_grid = cv2.bitwise_and(v, v, mask=mask_bean_inv)
    _, thresh_grid = cv2.threshold(v_grid, THRESHS['GRID'], 255, cv2.THRESH_BINARY)
    # 霍夫直線檢測
    lines = cv2.HoughLinesP(thresh_grid, **HOUGH_ARGS)
    img_grid_line = cv2.cvtColor(thresh_grid, cv2.COLOR_GRAY2BGR)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_grid_line, (x1,y1), (x2,y2), COLORS['LINE'], 2)
    mask_grid = cv2.inRange(img_grid_line, COLORS['LINE'], COLORS['LINE'])
    # 找格子 (輪廓) (算面積比例)
    contours_grid, _ = cv2.findContours(mask_grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_grid = list(contours_grid)
    # 刪除極端值格子
    # 上下限: Me ± IQR * ratio
    ratio = 0.5
    areas = np.array([cv2.contourArea(cnt) for cnt in contours_grid])
    me = np.median(areas)
    q1 = np.quantile(areas, q=0.25)
    q3 = np.quantile(areas, q=0.75)
    min = me - (q3 - q1) * ratio
    max = me + (q3 - q1) * ratio
    for i, area in list(enumerate(areas))[::-1]:
        if not min <= area <= max:
            contours_grid.pop(i)
            areas = np.delete(areas, i)
    cm2 = areas.mean()
    #
    img_grid_BGR = img_grid_line.copy()
    img_grid_bg = img_raw.copy()
    # 描繪輪廓
    cv2.drawContours(img_grid_BGR, contours_grid[:-1], -1, COLORS['CONTOUR'], 2)
    cv2.drawContours(img_grid_bg, contours_grid[:-1], -1, COLORS['CONTOUR'], 2)
    # 儲存
    write('grid.jpg', img_grid_BGR)
    write('grid_background.jpg', img_grid_bg)
    write('grid_value.jpg', v_grid)
    write('grid_thresh.jpg', thresh_grid)
    write('grid_line.jpg', img_grid_line)

    # 平方公分
    print(f'\033[A\r\033[J{cm2:,.2f} pixels is equal to 1 square centimeter', end='')
    results.insert(0, (0, cm2, 1, 'BENCHMARK'))
    # 
    del me, q1, q3, ratio, min, max, cm2
    del lines, contours_grid, areas
    del v_grid, thresh_grid, mask_grid
    del img_grid_line, img_grid_BGR, img_grid_bg
    #########################
    # Pyplot 圖表
    print(f'\033[2A\r\033[Jsaving {dataname}_plot.jpg')
    fig = plt.figure(figsize=(16,10), dpi=600, facecolor=COLORS['PLOT'])
    fig.suptitle(f'以{"純水" if dataname == "0" else f" {dataname} 食鹽水溶液"}種植', fontproperties=title_font)
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    grid = cv2.bitwise_and(img_raw, img_raw, mask=mask_bean_inv)
    bean = cv2.bitwise_and(img_raw, img_raw, mask=mask_bean)
    subplots = [
        ('原始照片', img_raw), ('綠豆', bean), ('格線', grid),
        ('色相(Hue)', h), ('飽和度(Saturation)', s), ('明度(Value)', v)
    ]
    for i, subplot in enumerate(subplots):
        plt.subplot(2, 3, i+1).set_title(subplot[0], fontproperties=font)
        plt.subplots_adjust(wspace=0.3, hspace=0.2)
        # Row 1
        if i < 3:
            plt.imshow(subplot[1], cmap='gray')
            continue
        # Row 2
        img = plt.imshow(subplot[1], vmin=0, vmax=255)
        if i == 3:
            img.set_cmap('hsv')
        bar = plt.colorbar(img, fraction=0.15, pad=0.1, orientation='horizontal')
        ticks = [i for i in range(0, 256, 16)]+[255]
        bar.set_ticks(ticks)
        bar.ax.set_xticklabels(ticks, rotation=315)
        bar.update_ticks()
    plt.savefig(path_plot)
    files.write(path_plot+'\n')
    # CSV
    print(f'\r\033[2Ksaving {dataname}_results.csv')
    with open(path_csv, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'pixel', 'cm2', 'STATUS'])
        writer.writerows(results)
    files.write(path_csv+'\n')
    #########################
    print('\033[2A\r\033[J', end='')

# 結束
end = time.time()
print(f'\r\033[JDONE! cost {end-start:.3f} seconds')
files.close()