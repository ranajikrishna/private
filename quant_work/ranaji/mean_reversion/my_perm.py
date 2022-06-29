
def perm(a,k=0):
   if(k==len(a)):
      return(a)
   else:
      for i in xrange(k,len(a)):
         a[k],a[i] = a[i],a[k]
         perm(a, k+1)
         a[k],a[i] = a[i],a[k]
