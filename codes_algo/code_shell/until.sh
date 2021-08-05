
# until loop in action
# Make sure we have space before and after + in expr
# Also the no-space before and after =
count=1
until [ $count -gt 10 ]
do
	echo $count
	count=`expr $count + 1`
done
