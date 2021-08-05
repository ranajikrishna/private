# To get out of a "case", we use ";;"
# The "?" is a substitution for one character,
# in contrast to "*" which is a substitution for all characters.
# ")" this is called a "parent".

echo "Enter a character:\c"
read var
case $var in 
[a-z])
	echo "Lower case"
	;;
[A-Z])
	echo "Upper case"
	;;
[0-9])
	echo "Digit"
	;;
?)
	echo "Special symbol"
	;;
*)
	echo "More than one character"
esac

