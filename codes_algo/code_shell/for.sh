
#For loop
# NOTE: The "*" means all the files in the home folder.

for item in *
do
	if [ -f $item ]
	then
		echo "$item"	
	fi
done
