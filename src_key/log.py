import sys

class Log:

    def __init__(self, stdout, filename):
        self.stdout = stdout
        self.logfile = open(filename, 'w')

    def write(self, text):
        self.stdout.write(text)
        self.logfile.write(text)

    def flush(self):
        self.stdout.flush()

    def close(self):
        self.stdout.close()
        self.logfile.close()

if __name__ == "__main__":
    log = Log(sys.stdout, 'log.txt')
    sys.stdout = log
    print("hello")