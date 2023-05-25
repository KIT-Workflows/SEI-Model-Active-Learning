import os, glob

files = 'db-non-vdw.yml'

if os.path.isfile(files):
    print("The file exists")
else:
    print("rendered_wano.yml added")

folder = 'DFT'
search_string = 'results.yml'

files = os.path.join("../" + folder, search_string)

print(files)