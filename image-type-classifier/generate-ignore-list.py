import os

print('hi')

items = []

for root, directories, files in os.walk(os.path.join(os.path.expanduser('~'), "Desktop", "Extras"), topdown=False):
    for name in files:
        items.append(name)

with open('files-to-ignore.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in items)
