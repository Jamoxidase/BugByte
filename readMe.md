I wrote this code to solve the BugByte graph puzzle [see here:](https://www.janestreet.com/bug-byte/)

This problem poses the question of whether or not a dynamic solution is necessary, given that this is just a one-off puzzle. 
This solution is based on a smart brute-force approach, where I first use the sum rule to determine viable edge weight permutations 
before considering the path rule, as the permutations constraint yields a superset of the path constraint solutions.

This is a pretty efficient approach, but the code could be refined. I may come back to this, but probably not.
