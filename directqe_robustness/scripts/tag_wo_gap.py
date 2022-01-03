import sys
argv=None
if argv is None:
	argv = sys.argv

tag_path = argv[1]
new_tag_path = argv[2]

def read_data(path):
	with open(path, 'r', encoding='utf-8') as f:
		data = f.read()
	return data.split('\n')

tag = read_data(tag_path)
new_tag = open(new_tag_path, 'w', encoding='utf-8')

for line in tag:
	line = line.split(' ')
	str = ''
	for i, token in enumerate(line):
		if i % 2 == 1:
			str = str + token + ' '
	new_tag.write(str.strip()+'\n')

new_tag.close()

# python scripts/tag_wo_gap.py /home/yanym/data_temp/task2/mello_version/dev.tag1 /home/yanym/data_temp/task2/mello_version/dev.tag