# 把mt+gap tag分离

tag_path = "/home/data_ti4_c/yanym/data/wmt20/ende/task2/mello_version_transquest/qe_test/test.mtgap_tag"
mt_tag_path = "/data1/yanym/data/temp/test.mt_tag"
gap_tag_path = "/data1/yanym/data/temp/test.gap_tag"

def read_data(path):
	with open(path, 'r', encoding='utf-8') as f:
		data = f.read()
	return data.split('\n')

tag = read_data(tag_path)
with open(mt_tag_path, 'w', encoding='utf-8') as fmt, open(gap_tag_path, 'w', encoding='utf-8') as fgap:
	for line in tag:
		line = line.split(' ')
		str = ''
		for i, token in enumerate(line):
			if i % 2 == 1:  # 仅保留mt
				str = str + token + ' '
		fmt.write(str.strip()+'\n')
		str = ''
		for i, token in enumerate(line):
			if i % 2 == 0:  # 仅保留gap
				str = str + token + ' '
		fgap.write(str.strip()+'\n')


# python mello_scripts/tag_wo_gap.py
