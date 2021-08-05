#NOTE: `` because commandlist is a textfile.
#NOTE: A ">>" is used to append the 
#helpfile (if we wanted to write to it we wou#ld have used ">")

#exec <<

for cmd in `cat < commandlist`
do
	echo "coz im happy"
	man $cmd >> helpfile
done

