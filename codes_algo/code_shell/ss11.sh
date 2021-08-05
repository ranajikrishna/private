#tput in action
tput clear
echo "Total no rows on screen = \c"
tput lines
echo "Total no of cols on screen =\c"
tput cols
tput cup 15 20
tput bold
echo "This should be in Bold"
echo "\033[0mBye"
