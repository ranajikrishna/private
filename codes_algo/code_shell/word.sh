# Note: An * in the end of [cond] matches the first character with the condition specifid. An * in the begining of the condition specified. 

echo "Enter a word:\c"
read word
case $word in
[aeiou]* | [AEIOU]*)
	echo "Begins with a Vowel"	
	;;
[0-9]*)
	echo "Begins with a Digit"	
	;;
*[0-9])
	echo "Ends with a Digit"
	;;
???)
	echo "You enetered a three letter word"
	;;
*)
	echo "Don't know what has been enetered"
	;;
esac

