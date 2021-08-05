
# Note: When we run the cmd "sh", shell open anot`her sub-shell. This sub-shell cannot access functions. SO what we do is we change mode to execution, i.e. "chmod +x funcExample.sh". Then we run the function script by cmd "$. funcExample.sh" (take note the the spacings). And the function is then finally run by typing their names, eg, "$youtube". If you want to remove the function from memory, type "unset youtube" 

youtube()
{
	echo "Good Morning"
}

byebye()
{
	cal
}
