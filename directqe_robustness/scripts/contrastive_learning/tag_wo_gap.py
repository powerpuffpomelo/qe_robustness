# 去掉 qe tag 的gap，也就是，保留偶数列

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


"""
python scripts/contrastive_learning/tag_wo_gap.py /home/data_ti4_c/yanym/data/wmt20/enzh/task2/mello_version_contrastive/qe_neg_out_of_order/train/train.mtgap_tag /home/yanym/data/train.mt_tag


"""