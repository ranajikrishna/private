# Check whether the input is a fine
# To check for others we use :-d for directories, -r for read files, -w for w# rite files, -x for executable files, -c for character files, -b for block f# iles (eg. image files), -s to check if file size is not 0.
 
echo "Enter a name:\c"
read fname
if [ -s $fname ] 
then
	echo "You indeed Entered a file name."
else
	echo "Wrong input"
fi
