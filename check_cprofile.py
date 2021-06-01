import pstats

fnum = input("Log file number:")
f = "tasks_"+str(fnum)+"time.log"
p = pstats.Stats(f)
p.sort_stats('tottime').print_stats(10)