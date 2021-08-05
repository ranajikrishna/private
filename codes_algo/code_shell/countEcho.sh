
echo "Enter the filename:\c"
read fname

terminal=`tty`

exec < $fname

nol=0
now=0
 nowl=0

while read line
do
	nol=`expr $nol + 1`
	nowl=`echo "$line" | wc -w`
	now=`expr $now + $nowl`
	#now=`expr $now + echo "$line" | wc -w` 

	#set $line
	#now=`expr $now + $#`
done

echo "No of lines:$nol"
echo "No of words:$now"

echo "yo"
exec < $terminal
