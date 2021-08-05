#escape sequence

echo "Hey world, nwhats up?"
echo "Hey world, \nwhats up?"
echo "I am a good hjkhjkhjkh  \rprogrammer"
echo "This would increase the \t\tspacing"
echo "Hey world, \b\b\bwhats up?"
# ----- These two dont work - They should give us statements in BOLD --
echo "\033[lmHey world whats up?"
echo "\033[lmHey world whats up?\033[0m"
