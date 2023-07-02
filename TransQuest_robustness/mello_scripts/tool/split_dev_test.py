#15.10
all_prefix = "/home/data_ti4_c/yanym/data/wmt20/ende/task2/dev/dev."
dev_prefix = "/home/yanym/temp/dev."
test_prefix = "/home/yanym/temp/test."

kinds = ['src', 'mt', 'source_tags', 'tags', 'hter', 'pe', 'src-mt.alignments']
for kind in kinds:
    with open(all_prefix + kind, 'r', encoding='utf-8') as fa, open(dev_prefix + kind, 'w', encoding='utf-8') as fd, \
        open(test_prefix + kind, 'w', encoding='utf-8') as ft:
        lines = fa.readlines()
        fd.writelines(lines[:500])
        ft.writelines(lines[500:])

# python /home/data_ti4_c/yanym/code/TransQuest/mello_scripts/split_dev_test.py