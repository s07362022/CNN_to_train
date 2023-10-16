import os
from traceback import print_tb

#資料夾的檔案
all_files = []
all_paths = []
all_files2 = []
all_paths2 = []

ss="F:\\nuck\\data\\data\\val\\img\\"
ss2 = "F:\\nuck\\data\\data\\val\\mask\\"
for path, dir, file in os.walk("F:\\nuck\\data\\data\\val\\img\\"): #資料夾路徑
    for f in file:
        # print(os.path.splitext(f)[-1])
        if os.path.splitext(f)[-1] in ['.png']:
            all_files.append(f)
            all_paths.append(path)
for path, dir, file in os.walk("F:\\nuck\\data\\data\\val\\mask\\"): #資料夾路徑
    for f in file:
        # print(os.path.splitext(f)[-1])
        if os.path.splitext(f)[-1] in ['.png']:
            all_files2.append(f)
            all_paths2.append(path)

#for i in range(len(all_files)):            
    #print(all_paths[i] + "\t" + all_files[i])

for i in range(len(all_files)):            
    print(all_paths[i]  + all_files[i])
#     k =0 #檔案名稱從1開始
    fname = all_files[i]
    fname2 = all_files2[i]
    # print()
    if fname==fname2:
        #print(fname.split('.')[0])
        new_fname = str(i)+".png"  #檔案名稱 "3_"+str(k)+ #fname.split('.')[0]+".png"
        #os.rename(os.path.join(all_paths[i], fname), os.path.join(all_paths[i], new_fname))
        os.rename(os.path.join(all_paths[i], fname), os.path.join(ss, new_fname))
        os.rename(os.path.join(all_paths2[i], fname), os.path.join(ss2, new_fname))