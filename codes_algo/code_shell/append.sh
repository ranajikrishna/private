#To change mode: chmod -w test (to remove write priveledges) , 
# chmod +w test (to add-on write priveledges).
# To write into the file directly, use "cat >> $fname".

echo "Enter a file name:\c"
read fname
if [ -f $fname ]
then
	if [ -w $fname ]
	then
		echo "Type matter to append. To quit press ctrl+d."
		cat >> $fname
	else
		echo "You do not have permission to write."
	fi
fi

