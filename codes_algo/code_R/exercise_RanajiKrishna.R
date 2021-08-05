
# Hi ALex, I have taken the liberty to complete the code the way I had initially set-out to do.
# I have created a simple test case for it ("Testing Data").
# The code could be improved upon by sorting the ids of "friend_remove" by column 1,
# and then by using binary serach to find the match. For an even higher efficiency, one could use 
# "Time Stamp" (not used here) to establish a starting point for searching in "friend_remove" (the reason being 
# that friends can be removed only after adding).

rm(list = ls(all = TRUE));  # clear all.
graphics.off();             # close all.

# ---- Testing Data ---
friend_accept <- matrix(0,10,3);              # Requests Accepted.       
friend_accept[,1] <- c(1,3,1,2,5,1,6,8,3,10);   
friend_accept[,2] <- c(2,9,8,7,1,3,7,9,1,4);    

friend_remove <- matrix(0,10,3);              # Requests Removed.
friend_remove[,1] <- c(1,13,1,7,5,9,6,3,12,3);
friend_remove[,2] <- c(4,4,8,2,1,9,8,1,10,9);
# ------

friendships <- matrix(0, 10,3);               # Create empty matrix for Friendships made.

size_accept <- dim(friend_accept)[1];
size_remove <- dim(friend_remove)[1];

itr1 <- 1;
itr2 <- 1;
itr3 <- 1;
incl <- 1;

while( itr1 <= size_accept ){                       # Iterate through Accept...
  
  while( itr2 <= size_remove ){                     # Iterate thorugh Remove...  
    if(friend_accept[itr1,1] == friend_remove[itr2,1] && friend_accept[itr1,2] == friend_remove[itr2,2]){ # Check if Pair in Accept appears in Remove
      incl <- 0;                                    # Set flag to: Don't add.
      itr2 <- itr2 + 1;
      break;
    }
    itr2 <- itr2 + 1;
  }
  if (incl ==1 ){                                   # Add if flag 1.
    friendships[itr3, ] <- friend_accept[itr1,];    # Populate Friendships.
    itr3 <- itr3 +1;
  }
  
  itr1 <- itr1 + 1; 
  itr2 <- 1;                            
  incl <- 1;        # Reset flag.

}

print(friendships)

