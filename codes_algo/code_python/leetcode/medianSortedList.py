
import pdb

class Solution:
    '''
    Given two sorted lists of different sizes, find the median of the combined
    list
    Explaination: https://www.youtube.com/watch?v=LPFhl65R7ww
    '''

    def median(self, list1: int, list2: int) -> float:
        len1 = len(list1)
        len2 = len(list2)

        if len1 > len2:
            list1, list2 = list2, list1
            len1, len2 = len2, len1

        total = len1+len2
#        half = (len1+len2-1)//2
        half = (len1+len2)//2 - 1

        list1_start = 0
        list1_end = len1-1 
#        list1_end = len1
        while True:
            idx1 = (list1_end - list1_start)//2
#            idx2 = half - idx1 - 1 
            idx2 = half - idx1

            list1_left = list1[idx1] if idx1>0 else -float('inf')
            list1_right = list1[idx1+1] if idx1+1<len1 else float('inf')
            list2_left = list2[idx2] if idx2>0 else -float('inf')
            list2_right = list2[idx2+1] if idx2+1<len2 else float('inf')
            

            if list1_left < list2_right and list2_left < list1_right :
                if total & 1:   # If `total` is ODD
                    return max(list1_left,list2_left)
                else:           # If `total` is EVEN
                    return (max(list1_left,list2_left) + \
                            min(list1_right,list2_right))/2
                return 

            elif list1_left > list2_right:
                list1_start -= 1

            else:
                list1_end += 1


def main():

    tmp = Solution()
    
    list1 = [1,5,6,7,12,14,15,16]
    list2 = [3,4,7,8,10,15,17,19]
    print(tmp.median(list1,list2),'\n')

    return 


if __name__ == '__main__':
    status = main()
    sys.exit()

