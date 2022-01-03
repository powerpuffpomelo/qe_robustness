bpe_file = "/data1/yanym/data/robust_qe_test/generate_dev/robust_test.mt.BPE"
bpe_id = "/data1/yanym/data/robust_qe_test/generate_dev/robust_test.idx.BPE"
original_id = "/data1/yanym/data/robust_qe_test/generate_dev/robust_test.idx"

with open(bpe_file, 'r', encoding='utf-8') as f1, open(bpe_id, 'r', encoding='utf-8') as f2, \
                                                    open(original_id, 'w', encoding='utf-8') as f3:
    for f1_line, f2_line in zip(f1, f2):
        origin_id_line = []
        bpe_file_line = f1_line.strip('\n').split()
        bpe_id_line = list(map(int, f2_line.strip('\n').split()))
        i = 0 # bpe
        j = 0 # origin
        for bpe_token in bpe_file_line:
            if i in bpe_id_line and str(j) not in origin_id_line:
                origin_id_line.append(str(j))
            if '@@' not in bpe_token:
                j += 1
            i += 1
        origin_id_line = ' '.join(origin_id_line) + '\n'
        f3.write(origin_id_line)

# python scripts/contrastive_learning/robust/id_bpe2origin.py