#Script to remove a given percent of samples from a text file
#samples are seperated by [SEP]
#usage: trim_text.py <filename> <percent> <new_filename>
#   where percent is a float from 0 to 1. 

import re
from sys import argv


#Parse arguments
if len(argv) < 3:
    SystemExit("usage: trim_text.py <filename> <percent> <new_filename>")
filename = argv[1]
percent = .5
try:
    percent = float(argv[2])
    if percent >= 1 or percent <= 0:
        SystemExit("percent must be a float between 0 and 1")
except ValueError:
    SystemExit("percent must be a float between 0 and 1")
new_filename = argv[3]

#Read file
text = ""
with open(filename) as fp:
  text = fp.read()
#print(text)

#Split into samples separtated by "[SEP]"
pattern = re.compile(".*\[SEP\]")
samples = re.findall(pattern, text)
#print(samples)

#Remove samples from back
num_to_remove = int(len(samples)*percent)
samples = samples[:-num_to_remove]
to_write = "\n".join(samples)

#Write to new file
with open(new_filename,"w") as fp:
    fp.write(to_write)



