import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--src_origin_path', type=str)
parser.add_argument('--tgt_origin_path', type=str)
parser.add_argument('--align_input_path', type=str)

args = parser.parse_args()

src_origin_path = args.src_origin_path
tgt_origin_path = args.tgt_origin_path
align_input_path = args.align_input_path

with open(src_origin_path, 'r', encoding='utf-8') as f1, open(tgt_origin_path, 'r', encoding='utf-8') as f2, \
    open(align_input_path, 'w', encoding='utf-8') as fs:
    src_lines = f1.readlines()
    tgt_lines = f2.readlines()
    for src_line, tgt_line in zip(src_lines, tgt_lines):
        new_line = src_line.strip('\n') + " ||| " + tgt_line
        fs.write(new_line)

# python scripts/contrastive_learning/contrastive_pretrain/awesome_align_input.py --src_origin_path $SRC_ORIGIN_PATH --tgt_origin_path $TGT_ORIGIN_PATH --align_input_path $ALIGN_INPUT_PATH