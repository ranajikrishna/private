
echo "Enter a filename: \c"
read fname
if [ -z "$fname" ]
then
	exit
fi

#terminal=`tty`

#Pass a filename from within the program
exec < $fname

count=1

while read line 
do
	echo $count.$line
	count=`expr $count + 1`
done

#exec < $terminal
