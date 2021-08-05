
echo "Enter a character:"
read var
if [ `echo $var | wc -c` -eq 2 ]
then
	echo "Correct!"
else
	echo "Wrong"
fi

#NOTE: 2 because there is a return charater
#appended to the output, as such the cursor appears
#on th enext line. Try this:
#read num
#408
#echo $num | wc -c
#This gives 4 as the result.



