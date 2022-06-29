
import numpy as np
import pdb


def is_palindrome(s):
    return s == s[::-1]

def main():
    s = "malayalam"
    ans = is_palindrome(s)

    if ans:
        print("Yes\n")
    else:
        print("No")
    return 

if __name__ == '__main__':
    status = main()
    sys.exit()

