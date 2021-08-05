# -i case insensitive
# -n the line number that has the word
# -c the frequency of the word
# -v the no other words. 

read $1 $2
grep -i -n -c $2 $1


