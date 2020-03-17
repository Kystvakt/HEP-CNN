#!/usr/bin/env python
import sys, os, time
import subprocess
import csv
import argparse

class SysStat:
    def __init__(self, procId, fileName=None, verbose=False):
        self.fileName = fileName
        self.verbose = verbose
        self.procdir = '/proc/%d' % procId

        self.pagesize = int(subprocess.check_output(['getconf', 'PAGESIZE']))
        self.nproc = int(subprocess.check_output(['nproc', '--all']))

        self.utime, self.stime, self.rss, self.vmsize = 0, 0, 0, 0
        self.io_read, self.io_write = 0, 0
        self.totaltime = 0

        #self.cpuFracs = []
        #self.readBytes = []
        #self.writeBytes = []
        #self.rsss = []

        if fileName != None:
            self.outFile = open(self.fileName, 'w')
            self.writer = csv.writer(self.outFile)
            columns = ["Datetime", "CPU", "RSS", 'VMSize', "Read", "Write", "Annotation"]
            self.writer.writerow(columns)

        self.update()

    def update(self, annotation=None):
        if not os.path.exists(self.procdir): return False
        if annotation == None: "''"

        with open(self.procdir+'/stat') as f:
            ## Note: see the linux man page proc(5)
            statcpu = f.read().split()
            utime = int(statcpu[13]) ## utime in jiffies unit
            stime = int(statcpu[14]) ## utime in jiffies unit
            vmsize = int(statcpu[22]) ## vm size in bytes
            rss   = int(statcpu[23])*self.pagesize ## rss in page size -> convert to bytes

        with open(self.procdir+'/io') as f:
            ## Note: /proc/PROC/io gives [rchar, wchar, syscr, syscw, read_bytes, write_bytes, cancelled_write_bytes]
            statio = f.readlines()
            io_read  = int(statio[4].rsplit(':', 1)[-1])
            io_write = int(statio[5].rsplit(':', 1)[-1])

        with open('/proc/stat') as f:
            stattotal = f.readlines()[0].split()
            totaltime = sum(int(x) for x in stattotal[1:]) ## cpu time in jiffies unit

        if self.totaltime != 0:
            cpuFrac   = 0 if totaltime == self.totaltime else 100*self.nproc*float( (utime-self.utime) + (stime-self.stime) ) / (totaltime-self.totaltime)
            readByte  = io_read-self.io_read
            writeByte = io_write-self.io_write

            #self.cpuFracs.append(cpuFrac)
            #self.readBytes.append(readByte)
            #self.writeBytes.append(writeByte)
            #self.rsss.append(rss)

            timestr = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
            stat = [timestr, cpuFrac, rss, vmsize, readByte, writeByte, annotation]
            if self.verbose: print(stat)
            if hasattr(self, 'writer'):
                self.writer.writerow(stat)
                self.outFile.flush()

        self.utime, self.stime, self.rss, self.vmsize = utime, stime, rss, vmsize
        self.io_read, self.io_write = io_read, io_write
        self.totaltime = totaltime

        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--interval', action='store', type=float, default=10, help='Time interval')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-o', '--output', action='store', type=str, default=None, help='Output file name')
    parser.add_argument('PID', action='store', type=int, help='Process ID to be monitored')
    args = parser.parse_args()

    procId = int(args.PID)

    sysstat = SysStat(procId, fileName=args.output, verbose=args.verbose)
    while sysstat.update(): time.sleep(args.interval)

