
import pandas as pd

def foo(df1, df2):

	df1['B'] = 1

	# change made in dev/home
	# change made in dev/work
	# NOTE: df1.join() returns a new dataframe. As such
	# df1= now holds the address of a new dataframe which is 
	# different from the dataframe that was passed to it.
	df1 = df1.join(df2['C'], how='inner', inplace)

	return()

def main(argv = None):

	df1 = pd.DataFrame(range(0,10,2), columns=['A'])
	df2 = pd.DataFrame(range(1,11,2), columns=['C'])

	foo(df1, df2)
	print df1

	return(0)
	

if __name__ == '__main__':
	status = main()
	sys.exit(status)
