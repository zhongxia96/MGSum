mode = ['train', 'valid']
path = '/home/jinhq/fairseq-master/relation_extraction/'
# for m in mode:
#     p = path + m + '.txt'
#     src_path = path + m + '.src'
#     tgt_path = path + m + '.tgt'
#     with open(p) as f:
#         con = f.readlines()
#     srcs = []
#     tgts = []
#     for i, line in enumerate(con):
#         split = line.split()
#         src = []
#         tgt = []
#         for s in split:
#             a, b = s.split('/')
#             a = a.split('_')
#             label = [b for _ in range(len(a))]
#             src = src + a
#             tgt = tgt + label
#         src = ' '.join(src)
#         tgt = ' '.join(tgt)
#         srcs.append(src)
#         tgts.append(tgt)
#
#     with open(src_path, 'w') as f:
#         for src in srcs:
#             f.write(src+'\n')
#
#     with open(tgt_path, 'w') as f:
#         for tgt in tgts:
#             f.write(tgt+'\n')


p = path + 'test' + '.txt'
src_path = path + 'test' + '.src'
tgt_path = path + 'test' + '.tgt'
with open(p) as f:
    con = f.readlines()
srcs = []
for i, line in enumerate(con):
    src = ' '.join(line.split('_'))
    srcs.append(src)

with open(src_path, 'w') as f:
    for src in srcs:
        f.write(src)

with open(tgt_path, 'w') as f:
    for tgt in srcs:
        f.write(tgt)

