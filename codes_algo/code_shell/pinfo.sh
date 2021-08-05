

echo "Enter username:\c"
read logname
line=`grep $logname /etc/passwd`
IFS=:
set $line
echo "Usename:$1"
echo "User ID:$3"
echo "GroupID:$4"
echo "Comment Filed:$5"
echo "Home Folder:$6"
echo "Home Default Shell: $7" 
