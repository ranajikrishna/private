
echo "Enter the Username to check:\c"
read logname

time=0 # Assumtion that the user has arady logged-in.

while true # Infinite loop (will keep executing)
do
	who | grep "$logname" > null
	if [ $? -eq 0 ]
	then
		echo "$logname has logged in"
		if [ $time -ne 0 ]
		then
			echo "$logname was $time minutes late in loggin in"
		fi 
		exit
	else
		time=`expr $time + 1`
		sleep 60	
	fi
done

