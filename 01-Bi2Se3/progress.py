from __future__ import print_function
# from time import sleep
import sys
import re


class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
                          r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')
        # sys.stdout.flush()
        # print('\r' + self.fmt % args, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)
        print('')
        # sys.stdout.flush()

if __name__ == "__main__":
    import time
    progress = ProgressBar(100, fmt=ProgressBar.FULL)
    for i in range(100):
        time.sleep(0.1)
        progress.current += 1
        progress()



