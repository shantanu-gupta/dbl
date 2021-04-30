import os
import sys

def read_trace():
    trace = []
    addresses = ['0x0000000000010000',
                 '0x0000000000020000',
                 '0x0000000000030000',
                 '0x0000000000040000',
                 '0x0000000000050000',
                 '0x0000000000060000']
    layer_in, addr_in = 0, None
    layer_out, addr_out = 0, None
    for line in sys.stdin:
        if line.startswith('Layer '):
            layer_out = int(line.split()[1])
            addr_in = addresses[layer_in]
            addr_out = addresses[layer_out]
            continue
        rw, ts, io = line.split()
        addr = addr_in if io == 'I' else addr_out
        rw = 'READ' if rw == 'R' else 'WRITE'
        ts = str(int(float(ts) * 1e6))
        trace.append((addr, rw, ts))
    return trace

def main():
    trace = read_trace()
    print('\n'.join(['\t'.join(t) for t in trace]))
    return

if __name__ == '__main__':
    main()
