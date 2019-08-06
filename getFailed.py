import glob as gl

all_files = gl.glob('*.log')
failed_lines = []

for fname in all_files:
    with open(fname, 'r') as f:
        lines = list(f)
    for l in lines:
        if 'failed' in l:
            failed_lines.append(l)

with open('failed.txt', 'w') as f:
    for l in failed_lines:
        f.write(l)
