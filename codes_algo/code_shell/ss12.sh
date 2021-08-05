#if-else statement in action
echo "Enter source and target file names."
read source target
if mv $source $target
then
echo "You file has been succesfully renamed."
fi
