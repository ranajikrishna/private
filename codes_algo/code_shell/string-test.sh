

# When comparing strings we use "=" and not "-eq", tha later is 
# used for mathematical expressions.
# The sign $? checks EXIT STATUS: 0 Correct, 1 In correct
# Unix stores a and b as string, so it wont make a difference if 
# we put "$a" or $a.

a=4.5
b=4.9

[ "$a" = "$b" ]
echo $?

