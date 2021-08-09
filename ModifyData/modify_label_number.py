import os

root = 'bikedata'
name = os.listdir(root)
for n in name:
    file_path = root + '/' + n
    new = ""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in lines:
            if i[0] == '0':
                x = i[0].replace('0', '5')
                i = x + i[1:]
            elif i[0] == '1':
                x = i[0].replace('1', '4')
                i = x + i[1:]
            elif i[0] == '2':
                x = i[0].replace('2', '3')
                i = x + i[1:]
            elif i[0] == '3':
                x = i[0].replace('3', '2')
                i = x + i[1:]
            elif i[0] == '4':
                x = i[0].replace('4', '1')
                i = x + i[1:]
            else:
                x = i[0].replace('5', '0')
                i = x + i[1:]

            new += i

    with open('renumber/' + n, 'w') as f:
        f.write(new)
