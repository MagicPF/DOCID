import os,stat
import urllib.request
filepath = 'data/Kyocera_CP1/train/OK/NewFile.txt'
f = open(filepath)
i = 0
while True:
    img_url = f.readline();
    if img_url == '':
        break
    try:
        with urllib.request.urlopen(img_url, timeout=30) as response, open('data/Kyocera_CP1/train/OK/'+str(i)+'.jpg', 'wb') as f_save:
            f_save.write(response.read())
            f_save.flush()
            f_save.close()
            print("成功")
        i = i + 1
    except:
        continue

f.close()