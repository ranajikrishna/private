
import sys
import pdb


def unique_watch(watch_time:list((int,int))) -> int:
    """
    Given a list of tuples representing start and stop times of video watching
    times, find the unique watching times of the video.
    """

    watch_time = sorted(watch_time)
    min_time = watch_time[0][0]
    idx, watch = 0, 0
    for item in watch_time[1:]:
        idx += 1 
        if item[1] <= watch_time[idx-1][1]:      # subset
            continue;
        elif(item[0] >= watch_time[idx-1][1]):   # disjoint
            watch += item[1] - item[0]
            min_time = item[0]
        else:                                    # intersect
            watch = item[1] - min_time
    return watch

def main():

    time = [(0,15),(10,250),(700,1000)]
    print(unique_watch(time))

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()

