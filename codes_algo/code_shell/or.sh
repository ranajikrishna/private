NOTE: There should be spaces as follows: $var = a -o $var = etc
echo "Enter a character:"
read var
if [ `echo $var | wc -c` -eq 2 ]
then
	if [ $var = a -o $var = e -o  $var = i -o $var = o -o  $var = u ]
	then
		echo "You entered a vowel"
else
		echo "You entered a consonant."
	fi
else 
	echo "You did not enter just 1 character."
fi




