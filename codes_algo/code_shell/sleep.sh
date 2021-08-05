
#echo "Enter a sentence:\c"
#read str

echo "Enter a filename:\c"
read fname

terminal=`tty`
exec < $fname

#for word in $str
#do
#	echo $word
#	sleep 1
#done

while read in line
do
	echo $line
	sleep 1
done

exec < $terminal
