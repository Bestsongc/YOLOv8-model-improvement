import os

labelfolder='D:/Download/project-11-at-2023-08-19-11-03-8229835f/labels'
imagesfolder='D:/Download/project-11-at-2023-08-19-11-03-8229835f/images'



lines=[]

def dell(file):

    file_name = file.strip(".txt")
    f=open(os.path.join(labelfolder, file), encoding='utf-8')# 打开当前txt标注文件
    conten=f.read(1)
    f.close()

    if conten=="1":
        os.remove(os.path.join(labelfolder, file))
        file_name=file_name+'.jpg'
        os.remove(os.path.join('D:/Download/project-11-at-2023-08-19-11-03-8229835f/images/', file_name))
    elif conten=="2":
        with open(os.path.join(labelfolder, file), 'r',encoding='utf-8') as f:
            lines = f.readlines()

        for i in range(len(lines)):
            if lines[i][0] == '2':
                lines[i] = '1' + lines[i][1:]
        with open(os.path.join(labelfolder, file), 'w', encoding='utf-8') as f:
            f.writelines(lines)



Filelist = os.listdir(labelfolder)

for a in Filelist:  # 这里的for循环开始依次对所有voc格式标注文件生成其对应的yolo格式的标注文件
    dell(a)